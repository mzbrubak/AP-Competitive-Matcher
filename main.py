import itertools
import math
from collections import defaultdict
from typing import Dict

import networkx

discouraged_games = defaultdict(lambda: 0)

# Thinker with these:

max_difference = 5  # The maximum skill difference between two players of the same game. Good value for two teams: 1. Good value for three teams: 2.
minimum_level = 2  # The minimum skill level of the worst player of a game. Good value teams: 3. At 5 teams or higher, you might need to lower the value to 2.
discouraged_games.update({  # Games that should be less likely to show up. This is a flat value added to error.
    "Stardew": 1,
    "VVVVVV": 1,
})
completely_disallowed_games = {  # Completely disallow these games. Doing so might increase performance over just setting a really high value above.
    "Slay the Spire",
    "ChecksFinder",
    "Clique",
}
discouraged_combinations = defaultdict(lambda: 0)
discouraged_combinations.update({
    # Example: ("Violet", "Timespinner"): 10,
})
disallowed_combinations = {
    # Example: ("Violet", "Stardew"),
}

# Turn this on for a performance increase (This will only consider the best match for each pair/trio/etc. of players):
# This means that "variants" of a player distribution (that only differ in the games that each pair plays) won't show up

only_use_best_match_for_player_combination = True

# Set the amount of teams. 7 is probably the max for reasonable computation time.:

teams = 3  # The max for this is probably 7.

# Determine how negative values are interpreted.
# A negative value means "I don't want to play this game but I will if I have to".
# a -5 is equivalent to a 5, but means they'd rather not play it this time around.

# 0 means their wishes will be entirely ignored and the game-player combination is considered fully
# -1 means that combination is now *banned* (eseentially: added to disallowed_combinations).
# Any positive number means this combination will be added to disallowed_combinations with the specified value.

negative_entry_treatment = 10

# Force two players to be on the same team. Only supports duos at the moment (You can chain them to simulate trios).
# (This uses some hacky Python lol)

force_same_team = {
    # Example: ("Violet", "Dragorrod"),
}

# Disallow two players from being on the same team. Only supports duos (You can chain them to simulate trios).

force_different_team = {
    # Example: ("Violet", "Dragorrod"),
}

# Finally, tinker with the value function:


def get_compatibility_score(a: int, b: int):
    return abs(a - b)*teams + (5 - min(a, b))**2


# Don't touch anything past here unless you know what you're doing!

results = []
achievable_score = math.inf
results_amount = 10


def get_cum_compatibility_score(scores):
    return sum(get_compatibility_score(c[0], c[1]) for c in itertools.combinations(scores, 2)) / (binom(teams, 2))


class Person:
    name: str = ""
    games: Dict[str, int] = dict()

    def __init__(self):
        self.games = dict()

    def get_overlap(self, other):
        best = (math.inf, "")

        for game, this_score in self.games.items():
            if game not in other.games:
                continue

            if (self.name, game) in disallowed_combinations or (other.name, game) in disallowed_combinations:
                continue

            if abs(this_score - other.games[game]) > max_difference:
                continue

            if this_score < minimum_level or other.games[game] < minimum_level:
                continue

            new_candidate = get_compatibility_score(this_score, other.games[game]) + get_discouragement_factor(game, [self, other])
            if best[0] > new_candidate:
                best = (new_candidate, game)

        return best


def balance_teams(result):
    people_to_games = dict()
    for game_and_players in result[0]:
        for player in game_and_players[0]:
            people_to_games[player] = game_and_players[1]

    team_dist_list = [[[(player, player.games[result[0][0][1]])] for player in result[0][0][0]]]
    for game_and_players in result[0][1:]:
        new_team_dist_list = []
        for current_team_dist in team_dist_list:
            for new_players in itertools.permutations([(player, game_and_players[1]) for player in game_and_players[0]]):
                new_team_dist = []
                for i, player in enumerate(new_players):
                    player_with_score = (player[0], player[0].games[player[1]])
                    team = current_team_dist[i] + [player_with_score]
                    new_team_dist.append(team)
                new_team_dist_list.append(new_team_dist)
        team_dist_list = new_team_dist_list

    team_possibilites_with_scores = []
    for possibility in team_dist_list:
        wrong = False

        if force_same_team:
            for forced_teammates in force_same_team:
                forced_teammates = set(forced_teammates)
                for team in possibility:
                    team_set = set(player.name for player, _ in team)
                    if not forced_teammates.isdisjoint(team_set) and not forced_teammates.issubset(team_set):
                        print(team_set)
                        print(forced_teammates)
                        wrong = True
                        break

                if wrong:
                    break

        if force_different_team:
            for disallowed_teammates in force_different_team:
                disallowed_teammates = set(disallowed_teammates)
                for team in possibility:
                    team_set = set(player.name for player, _ in team)
                    if disallowed_teammates.issubset(team_set):
                        wrong = True
                        break
                if wrong:
                    break

        if wrong:
            continue

        team_possibilites_with_scores.append([(team, sum(player[1] for player in team)) for team in possibility])

    best = min(team_possibilites_with_scores, key=lambda possibility: max(team[1] for team in possibility) - min(team[1] for team in possibility))

    print("Best teams:")
    for i, team in enumerate(best):
        print("Team " + str(i + 1) + ": " + ", ".join([player[0].name + f" on {people_to_games[player[0]]} ({player[1]})" for player in team[0]]) + " | Score: " + str(sum([player[1] for player in team[0]])))


def print_combination(arr, n, r):
    data = [0] * r

    combination_util(arr, data, 0,
                     n - 1, 0, r)


def get_score(cycles):
    return sum(cycle[2] for cycle in cycles)


def binom(n, k):
    return math.factorial(n) // math.factorial(k) // math.factorial(n - k)


def print_result(cycles):
    global results
    global achievable_score

    new_score = get_score(cycles)

    if len(results) < results_amount:
        results.append((cycles.copy(), new_score))
        return

    results_a = results

    if new_score < results[-1][1]:
        results.pop()
        results.append((cycles.copy(), new_score))
        results.sort(key=lambda r: r[1])
        achievable_score = results[-1][1]
        return


def combination_util(arr, data, start,
                     end, index, r):
    global achievable_score

    if index == r:
        print_result(data)
        return

    i = start

    ppl_so_far = {item.name for sublist in data if sublist for item in sublist[0]}

    while i <= end and end - i + 1 >= r - index:
        ppl_from_new_cycle = {person.name for person in arr[i][0]}

        if ppl_so_far & ppl_from_new_cycle:
            i += 1
            continue

        data[index] = arr[i]

        if get_score([x for x in data if x]) > achievable_score:
            i += 1
            continue

        combination_util(arr, data.copy(), i + 1,
                         end, index + 1, r)
        i += 1


def find_cycle_set(cycles, n):
    combinations = []
    print_combination(cycles, len(cycles), n)


def get_discouragement_factor(game, people):
    return discouraged_games[game] + sum(discouraged_combinations[(person.name, game)] for person in people)


def n_matching_experimental(persons, games):
    possible_tuples = []

    for game in games:
        associated_persons = [person for person in persons if game in person.games]

        new_tuples = [(combination, game, get_cum_compatibility_score(person.games[game] for person in combination) + get_discouragement_factor(game, combination))
                      for combination in itertools.combinations(associated_persons, teams)]
        valid_tuples = []

        for tuple in new_tuples:
            if tuple[1] in completely_disallowed_games:
                continue

            if any((person.name, tuple[1]) in disallowed_combinations for person in tuple[0]):
                continue

            problem = False

            for combination in itertools.combinations(tuple[0], 2):
                person_a = combination[0]
                person_b = combination[1]
                score_a = person_a.games[tuple[1]]
                score_b = person_b.games[tuple[1]]

                if abs(score_a - score_b) > max_difference:
                    problem = True
                    break

                if score_a < minimum_level or score_b < minimum_level:
                    problem = True
                    break

                if (person_a.name, person_b.name) in force_same_team or (person_b.name, person_a.name) in force_same_team:
                    problem = True
                    break

            if problem:
                continue

            equivalent_tuple = [tuple2 for tuple2 in possible_tuples if all(person in tuple2[0] for person in tuple[0])]

            if only_use_best_match_for_player_combination:
                if equivalent_tuple:
                    if tuple[2] < equivalent_tuple[0][2]:
                        possible_tuples.remove(equivalent_tuple[0])
                        valid_tuples.append(tuple)
                        continue
                    else:
                        continue

            valid_tuples.append(tuple)

        possible_tuples += valid_tuples

    find_cycle_set(possible_tuples, int(len(persons) / teams))

    global results
    results.sort(key=lambda x: x[1])

    for result in results:
        for game in result[0]:
            print(", ".join([person.name for person in game[0]]) + " playing " + game[1] + ". Compatibility error: " + str(round(game[2])))
        print("Overall score: " + str(round(result[1])))

        balance_teams(result)

        print("---")


if __name__ == '__main__':
    if not disallowed_combinations:
        disallowed_combinations = set()
    if not force_same_team:
        force_same_team = set()
    if not force_different_team:
        force_different_team = set()

    persons = []

    with open("values.txt", "r") as file:
        game_names = dict()

        for line in file.readlines():
            line_real = line.strip()
            line_split = line_real.split("\t")
            if not line_split:
                continue
            if line_split[0] == "Discord ID":
                for i, value in enumerate(line_split):
                    if i == 0:
                        continue
                    game_names[i] = value
                continue

            new_person = Person()
            new_person.name = line_split[0]

            for i, value in enumerate(line_split):
                if i == 0:
                    continue
                if value == "":
                    continue

                value = int(value)

                if value < 0:
                    value = abs(value)

                    if negative_entry_treatment == -1:
                        disallowed_combinations.add((new_person.name, game_names[i]))

                    if negative_entry_treatment > 0:
                        discouraged_combinations[(new_person.name, game_names[i])] = negative_entry_treatment

                new_person.games[game_names[i]] = value

            persons.append(new_person)

    if teams == 2:
        print("For team size 2, the regular algorithm might take very long to complete. But, 2-matching is actually a solvable (P) problem where it is easy to get the singular best answer. That answer is this:")
        print("")
        G = networkx.Graph()
        G.add_nodes_from([person for person in persons])

        for i, person_a in enumerate(persons):
            for j, person_b in enumerate(persons):
                if i >= j:
                    continue

                score, game_name = person_a.get_overlap(person_b)

                if game_name in completely_disallowed_games:
                    continue

                if score == math.inf:
                    continue

                G.add_edge(person_a, person_b, weight=score)

        if teams == 2:
            best_matching = networkx.min_weight_matching(G)

            cum_sum = 0

            for person_a, person_b in best_matching:
                score, game = person_a.get_overlap(person_b)
                score_a = person_a.games[game]
                score_b = person_b.games[game]

                cum_sum += get_compatibility_score(score_a, score_b) + get_discouragement_factor(game, [person_a, person_b])

                favored_person = person_a.name if score_a > score_b else (
                    person_b.name if score_b > score_a else "neither player")

                print(
                    f"{person_a.name} and {person_b.name}, playing {game}. This matchup favors {favored_person}, ({score_a},{score_b}).")
        print("\nThe regular algorithm will now also be performed. Be aware this might take minutes, if not hours, with a player count of 18 or higher.")

    if teams >= 3:
        print("Please be aware that this problem is NP-complete. This means that its execution time grows exponentially. With over 20 players, you might have over a minute, if not several.")

    print("You can always try setting the values for minimum skill and maximum skill difference to be more restrictive.\nIf that doesn't work, you could try pre-setting some match-ups and removing those players from values.txt to compute a solution for the rest of the players, then combining your pre-set matchup with those results.")
    print("---")

    n_matching_experimental(persons, game_names.values())
