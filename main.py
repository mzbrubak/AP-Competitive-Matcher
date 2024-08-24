#import copy
import itertools
import math
#import time # for performance tests
from collections import defaultdict
from typing import Dict

import networkx
import sys

discouraged_games = defaultdict(lambda: 0)

# Tinker with these:

max_difference = 3  # The maximum skill difference between two players of the same game. Good value for two teams: 1. Good value for three teams: 2.
minimum_level = 2 # The minimum skill level of the worst player of a game. Good value teams: 3. At 5 teams or higher, you might need to lower the value to 2.
lowered_minimum = 2 # If there is a player that cannot match with anyone in the current settings, lower standard *for them* to this value.
# Recommended action for restrictive games is to alternate lowering minimum_level and lowered_minimum.

discouraged_games.update({  # Games that should be less likely to show up. This is a flat value added to error.
    "Stardew": 1,
    "VVVVVV": 1,
})
completely_disallowed_games = {  # Completely disallow these games. Doing so might increase performance over just setting a really high value above.
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

teams = 0  # The max for this is probably 7.

# Determine how negative values are interpreted.
# A negative value means "I don't want to play this game but I will if I have to".
# a -5 is equivalent to a 5, but means they'd rather not play it this time around.

# 0 means their wishes will be entirely ignored and the game-player combination is considered fully
# -1 means that game-player combination is now *banned* (eseentially: added to disallowed_combinations).
# Any positive number means this game-player combination will be added to discouraged_combinations with that value.

negative_entry_treatment = -1

# Force two players to play different worlds.
# This can be used to force two players on the same team, but you'll have to balance the teams yourself.
# Only supports duos at the moment (You can chain them to simulate trios). (This uses some hacky Python lol)

force_different_game = {
    # Example: ("Violet", "Dragorrod"),
}

# Disallow two players from being on the same team. Only supports duos (You can chain them to simulate trios).

force_different_team = {
    # Example: ("Violet", "Dragorrod"),
}


# Will print team combos immediately as they are found. This helps with getting ANY result on a big player count.
# These will look very similar for some time, it will appear to get "stuck" on some idea.

print_results_immediately = False


# If set to True, will use a brute force algorithm to get the best possible team balancing.
# If set to False, will use a greedy algorithm to make the best of what it can.

perfect_team_balancing = False

# Amount of results to print in the end.
# Setting this lower can actually improve performance, as there is node pruning by "minimum achievable score".
# It only improves performance if any combinations are being found, though.
# I.e.: Set this to 1 if you just want the best combination as quickly as possible. Otherwise, probably leave it at 10.

results_amount = 1


# Finally, tinker with the value function:

def get_compatibility_score(a: int, b: int):
    return abs(a - b)*teams + (4 - min(4,a, b))**2


# Don't touch anything past here unless you know what you're doing!

results = []
achievable_score = math.inf
worst_player_count = math.inf


def get_cum_compatibility_score(scores,size):
    return sum(get_compatibility_score(c[0], c[1]) for c in itertools.combinations(scores, 2)) / (binom(size, 2))

class Person:
    name: str = ""
    games: Dict[str, int] = dict()

    def __init__(self):
        self.games = dict()

    def get_overlap(self, other):
        best = (math.inf, "")

        for game, this_score in self.games.items():
        
            if game in completely_disallowed_games:
                continue
                
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


def find_better_dist(team_dist):
    scores = [sum(player_and_score[1] for player_and_score in team) for team in team_dist]
    avg_score = sum(scores) / len(scores)
    error = sum(abs(score - avg_score) for score in scores)

    for team1 in range(0, len(team_dist)):
        for team2 in range(0, len(team_dist)):
            for player1 in range(0, len(team_dist[0])):
                swapped_team_dist = [team.copy() for team in team_dist]
                temp = swapped_team_dist[team1][player1]
                swapped_team_dist[team1][player1] = swapped_team_dist[team2][player1]
                swapped_team_dist[team2][player1] = temp

                new_scores = [sum(player_score[1] for player_score in team) for team in swapped_team_dist]
                new_error = sum(abs(score - avg_score) for score in new_scores)

                if new_error < error:
                    return swapped_team_dist, True

    return team_dist, False


def balance_teams(result):
    people_to_games = dict()
    for game_and_players in result[0]:
        for player in game_and_players[0]:
            people_to_games[player] = game_and_players[1]

    if perfect_team_balancing:
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

            if force_different_game:
                for forced_teammates in force_different_game:
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

    else:
        team_dist = [[] for i in range(0, len(result[0][0][0]))]
        for game_and_players in result[0]:
            for index, player in enumerate(game_and_players[0]):
                team_dist[index].append((player, player.games[game_and_players[1]]))

        for _ in range(0, 100):
            new_dist, found_better = find_better_dist(team_dist)

            if not found_better:
                break

            team_dist = new_dist

        best = [(team,) for team in team_dist]

    print("Best teams:")
    for i, team in enumerate(best):
        print("Team "
              + str(i + 1)
              + ": "
              + ", ".join([player[0].name + f" on {people_to_games[player[0]]} ({player[1]})" for player in team[0]])
              + " | Score: " + str(sum([player[1] for player in team[0]])))


def find_cycle_set(arr, r):
    n=len(arr)
    data = [0] * r
    combination_util(arr, data, 0,
                     n - 1, 0, r)
    


def get_score(cycles):
    return sum(cycle[2] for cycle in cycles)


def binom(n, k):
    return math.factorial(n) // math.factorial(k) // math.factorial(n - k)


def print_single_result(result):
    for game in result[0]:
        print(", ".join([person.name for person in game[0]]) + " playing " + game[
            1] + ". Compatibility error: " + str(round(game[2])))
    print("Overall score: " + str(round(result[1])))
    if teams!=0:#balancing doesn't work if teams are different sizes
        balance_teams(result)

    print("---")


def print_result(cycles):
    global results
    global achievable_score
    new_score = get_score(cycles)

    #add new result if there aren't enough results yet
    if len(results) < results_amount:
        if not len(results):#i.e. if this is the first result
            print("Found something! If things are taking too long, rerun with 'print_results_immediately = True'.")
        results.append((cycles.copy(), new_score))
        if print_results_immediately:
            print_single_result((cycles.copy(), new_score))
        return

    #add new result if it's better than the worst result
    if new_score < results[-1][1]:
        results.pop()
        results.append((cycles.copy(), new_score))
        if print_results_immediately:
            print_single_result((cycles.copy(), new_score))
        results.sort(key=lambda r: r[1])
        achievable_score = results[-1][1]#so achievable score is the worst score currently kept
        return


def combination_util(arr, data, start,
                     end, index, r):
    global achievable_score

    #check if enough matches have been made, print results if so
    if index == r:
        print_result(data)
        return

    #filter remaining tuples to remove any that contain players already matched.  Return if there are none
    ppl_so_far = {item for sublist in data if sublist for item in sublist[0]}
    remaining_tuples = {tuple[0] for tuple in arr[start:] if not set(tuple[0]) & ppl_so_far}
    if not remaining_tuples:
        return

    #check if possible matches still exist for all remaining players, return if not (note: for pairing mode I'll need to change how remaining_people is checked, and possibly )
    if data[0]:
        remaining_people = teams * sum(1 for d in data if not d)
        ppl_in_remaining_tuples = {j for sub in remaining_tuples for j in sub}    
        if remaining_people != len(ppl_in_remaining_tuples):
            return

    #if all previous checks pass, iterate over all remaining tuples
    i = start
    while i <= end and end - i + 1 >= r - index:#second condition triggers if not enough tuples are left for full game
        if index == 0:
            print(f"{i+1}/{worst_player_count}. Later iterations go faster.")

        #skip tuples removed because players were already matched
        if arr[i][0] not in remaining_tuples:
            i += 1
            continue

        #skip tuples that force the score to be worse than the worst kept combination
        data[index] = arr[i]
        if get_score([x for x in data if x]) > achievable_score:
            i += 1
            continue
        
        #if tuple is good, temporarily commit to it and look for next one
        combination_util(arr, data.copy(), i + 1,
                         end, index + 1, r)
        
        #once recursive function returns, move on to next tuple
        i += 1

def combination_util_partner(arr3, arr2, data, start,
                     end, index, r):
    global achievable_score
    #check if enough matches have been made, print results if so
    if index == r[0] + r[1]:
        print_result(data)
        return

    #filter remaining tuples to remove any that contain players already matched.  Return if there are none
    ppl_so_far = {item for sublist in data if sublist for item in sublist[0]}
    remaining_triples = {tuple[0] for tuple in arr3[start:] if not set(tuple[0]) & ppl_so_far}
    remaining_doubles = {tuple[0] for tuple in arr2[start:] if not set(tuple[0]) & ppl_so_far}
    if not remaining_doubles or (index<r[0] and not remaining_triples):
        return

    #check if possible matches still exist for all remaining players, return if not (note: for pairing mode I'll need to change how remaining_people is checked, and possibly )
    if data[0]:
        remaining_people = 2*(r[0]+r[1]-index) + max(0,r[0]-index)
        ppl_in_remaining_doubles = {j for sub in remaining_doubles for j in sub}#for every remaining triple, there are three remaining doubles, so this is enough
        if remaining_people != len(ppl_in_remaining_doubles):
            return


    #need to separately check if there are enough triplets
    if index>r[0]:
        arr=arr2
        i=start
        remaining_tuples=remaining_doubles
        req=r[1]+r[0]
    elif index==r[0]:
        arr=arr2
        i=1
        end=len(arr2)-1
        remaining_tuples=remaining_doubles   
        req=r[1]+r[0]     
    else:
        arr=arr3
        i=start
        remaining_tuples=remaining_triples
        req=r[0]


    #if all previous checks pass, iterate over all remaining tuples

    while i <= end and end - i + 1 >= req - index:#second condition triggers if not enough tuples are left for full game
        if index == 0:
            print(f"{i+1}/{worst_player_count}. Later iterations go faster.")

        #skip tuples removed because players were already matched
        if arr[i][0] not in remaining_tuples:
            i += 1
            continue

        #skip tuples that force the score to be worse than the worst kept combination
        data[index] = arr[i]
        if get_score([x for x in data if x]) > achievable_score:
            i += 1
            continue
        
        #if tuple is good, temporarily commit to it and look for next one
        combination_util_partner(arr3, arr2, data.copy(), i + 1,
                         end, index + 1, r)
        
        #once recursive function returns, move on to next tuple
        i += 1

def get_discouragement_factor(game, people):
    return discouraged_games[game] + sum(discouraged_combinations[(person.name, game)] for person in people)


def generate_tuples(persons, games, size, problematic_players=frozenset()):
    possible_tuples = []

    for game in games:
        associated_persons = [person for person in persons if game in person.games]

        new_tuples = [(combination, game, get_cum_compatibility_score((person.games[game] for person in combination),size) + get_discouragement_factor(game, combination))
                      for combination in itertools.combinations(associated_persons, size)]

        valid_tuples = []

        for tuple in new_tuples:
            if tuple[1] in completely_disallowed_games:
                continue#note: consider performance cost of checking disallowed here or before finding tuples?

            if any((person.name, tuple[1]) in disallowed_combinations for person in tuple[0]):
                continue

            problem = False

            minimum_to_use = minimum_level

            if set(tuple[0]) & problematic_players:
                minimum_to_use = lowered_minimum

            for combination in itertools.combinations(tuple[0], 2):
                person_a = combination[0]
                person_b = combination[1]
                score_a = person_a.games[tuple[1]]
                score_b = person_b.games[tuple[1]]

                if abs(score_a - score_b) > max_difference:
                    problem = True
                    break

                if score_a < minimum_to_use or score_b < minimum_to_use:
                    problem = True
                    break

                if (person_a.name, person_b.name) in force_different_game or (person_b.name, person_a.name) in force_different_game:
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

    return possible_tuples


def n_matching_experimental(persons, games):
    global worst_player_count

    possible_tuples = generate_tuples(persons, games, teams)

    too_restrictive_players = set()

    for player in persons:
        if not any(player in tuple[0] for tuple in possible_tuples):
            too_restrictive_players.add(player)
            print(f"With these settings, player {player.name} does not play any game that {teams - 1} other players play.")

    if too_restrictive_players and minimum_level != lowered_minimum:
        print("As a result, no combinations were found.")
        print("Attempting to lower standard.")

        possible_tuples = generate_tuples(persons, games, teams, frozenset(too_restrictive_players), )

        #print("Make the restrictions looser, or play with less teams.")

        too_restrictive_players = set()

        for player in persons:
            if not any(player in tuple[0] for tuple in possible_tuples):
                too_restrictive_players.add(player)
                print(f"Even with the lowered standard, player {player.name} does not play any game that {teams - 1} other players play.")

    all_t = len(possible_tuples)

    counts = dict()

    for tuple in possible_tuples:
        for player in tuple[0]:
            counts.setdefault(player, 0)
            counts[player] += 1

    worst_player_count = min(counts.values())

    possible_tuples.sort(key=lambda t: sum(counts[p]*pow(all_t, -ind) for ind, p in enumerate(sorted(t[0], key=lambda p: counts[p]))))

    find_cycle_set(possible_tuples, int(len(persons) / teams))

    global results
    results.sort(key=lambda x: x[1])

    if not results:
        print("No combinations were found.")
        print("Make the restrictions looser, or play with less teams.")

    for result in results:
        print_single_result(result)

def partner_matching_experimental(persons,games):
    global worst_player_count
    global teams
    playercount=len(persons)
    if playercount<3:
        sys.exit("Not enough players.  If there's only two of you, you really don't need an algorithm to pick a game.  If there's only one of you, invite someone else to race you!")
    playerremainder=playercount%4
    if playerremainder:#if 0, just run teams=2 and split manually
        optimaltriadcount=(2-playerremainder)%4+2 #maps 1 to 3, 2 to 2, and 3 to 5
        if playercount<optimaltriadcount*3:
            print("For certain low playercounts (3,5,7, and 11) it is not possible to generate a set of games where everyone is playing a competitive (more than 1 team) multiworld (more than one slot per team).  Matching will proceed while generating some competitive single-world matches")
            optimaltriadcount=1
        pairplayercount=playercount-3*optimaltriadcount
        possible_triples=generate_tuples(persons,game_names.values(),3)
        possible_doubles=generate_tuples(persons,game_names.values(),2)
        too_restrictive_players3=set()
        too_restrictive_players2=set()
        for player in persons:
            if not any(player in double[0] for double in possible_doubles):
                too_restrictive_players2.add(player)
                print(f"With these settings, player {player.name} does not play any game that any other player plays.")
            if not any(player in triple[0] for triple in possible_triples):
                too_restrictive_players3.add(player)
        if len(too_restrictive_players3)>pairplayercount:
            print(f"There are ", len(too_restrictive_players3), " players that do not play any games that two other players play")
        if (too_restrictive_players2 or (len(too_restrictive_players3)>pairplayercount)) and minimum_level!=lowered_minimum:
            print("As a result, no combinations exist.  Attempting to lower standard")
            possible_triples=generate_tuples(persons,game_names.values(),3,frozenset(too_restrictive_players2|too_restrictive_players3))
            possible_doubles=generate_tuples(persons,game_names.values(),2,frozenset(too_restrictive_players2|too_restrictive_players3))
            too_restrictive_players3=set()
            too_restrictive_players2=set()
            for player in persons:
                if not any(player in double[0] for double in possible_doubles):
                    too_restrictive_players2.add(player)
                    print(f"Even with the lowered standard, player {player.name} does not play any game that any other player plays.")
                if not any(player in triple[0] for triple in possible_triples):
                    too_restrictive_players3.add(player)
            if len(too_restrictive_players3)>pairplayercount:
                print(f"Even with the lowered standard, there are ", len(too_restrictive_players3), " players that do not play any games that two other players play")
#all_t and worst_player_count are just for logging progress I think, basing it on the outer loop should be fine?
        all_t3=len(possible_triples)
        all_t2=len(possible_doubles)
        counts2=dict()
        for tuple in possible_doubles:
            for player in tuple[0]:
                counts2.setdefault(player, 0)
                counts2[player] += 1
        counts3=dict()
        for tuple in possible_triples:
            for player in tuple[0]:
                counts3.setdefault(player, 0)
                counts3[player] += 1
        worst_player_count=min(counts3.values())
        possible_doubles.sort(key=lambda t: sum(counts2[p]*pow(all_t2, -ind) for ind, p in enumerate(sorted(t[0], key=lambda p: counts2[p]))))
        possible_triples.sort(key=lambda t: sum(counts3[p]*pow(all_t3, -ind) for ind, p in enumerate(sorted(t[0], key=lambda p: counts3[p]))))
        combination_util_partner(possible_triples,possible_doubles, [0]*int(optimaltriadcount+(pairplayercount/2)),0,len(possible_triples)-1,0,[optimaltriadcount,pairplayercount/2])
        global results
        results.sort(key=lambda x: x[1])

        if not results:
            print("No combinations were found. Find more people or loosen the restrictions.")

        for result in results:
            print_single_result(result)
    else:
        print("Player count is divisible by 4, so running algorithm for teams=2 is sufficient for now.")
        teams=2
        print("For team size 2, the regular algorithm might take very long to complete. But, 2-matching is actually a solvable (P) problem where it is easy to get the singular best answer. That answer is this:")
        print("")
        optimal_2team_matching(persons)
        print("\nThe regular algorithm will now also be performed. Be aware this might take minutes, if not hours, with a player count of 18 or higher.")
        n_matching_experimental(persons, game_names.values())

def optimal_2team_matching(persons):
    G = networkx.Graph()
    G.add_nodes_from([person for person in persons])

    for i, person_a in enumerate(persons):
        for j, person_b in enumerate(persons):
            if i >= j:
                continue

            score, game_name = person_a.get_overlap(person_b)

            if score == math.inf:
                continue

            G.add_edge(person_a, person_b, weight=score)


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

#tstart=time.time()
if __name__ == '__main__':
    if not disallowed_combinations:
        disallowed_combinations = set()
    if not force_different_game:
        force_different_game = set()
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
        optimal_2team_matching(persons)
        print("\nThe regular algorithm will now also be performed. Be aware this might take minutes, if not hours, with a player count of 18 or higher.")
        n_matching_experimental(persons, game_names.values())

    if teams >= 3:
        print("Please be aware that this problem is NP-complete. This means that its execution time grows exponentially. With over 20 players, you might have over a minute, if not several.")
        n_matching_experimental(persons, game_names.values())

    if teams == 0: #small games/partner mode: decompose group automatically into 2v2 matches, with an additional 3-team match with up to 5 players per team as required to allow all players to join regardless of number
        print("Please note that this is an experimental matching mode.")
        partner_matching_experimental(persons,game_names.values())



#tend=time.time()
#print(str(tend-tstart))