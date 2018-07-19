import os
import random
import math

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker

PIECE_X = "X"
PIECE_O = "O"
PIECE_EMPTY = " "

STATE_NOT_FINISHED = "not finished"
STATE_WIN_X = "X wins"
STATE_WIN_O = "O wins"
STATE_DRAW = "draw"


class XOBoard:

  def __init__(self):
    self._board_state = [
      [PIECE_EMPTY, PIECE_EMPTY, PIECE_EMPTY],
      [PIECE_EMPTY, PIECE_EMPTY, PIECE_EMPTY],
      [PIECE_EMPTY, PIECE_EMPTY, PIECE_EMPTY],
    ]

    self._current_turn = PIECE_X

  def place(self, row, col):
    self._place_board_piece(row, col)

    if self._current_turn == PIECE_X:
      self._current_turn = PIECE_O
    else:
      self._current_turn = PIECE_X

  def get_game_lifecycle_state(self):
    if self._player_has_won(PIECE_X):
      return STATE_WIN_X

    if self._player_has_won(PIECE_O):
      return STATE_WIN_O

    if True in [PIECE_EMPTY in row for row in self._board_state]:
      return STATE_NOT_FINISHED

    return STATE_DRAW

  def get_available_actions(self):
    available_actions = []
    for row in range(0, 3):
      for col in range(0, 3):
        if self._board_state[row][col] == PIECE_EMPTY:
          available_actions.append([row, col])

    return available_actions

  def mock_next_board_state(self, action):
    row, col = action

    self._place_board_piece(row, col)
    mock_board_state = self._stringify_board_state()
    self._board_state[row][col] = PIECE_EMPTY

    return mock_board_state

  def get_immutable_board_state(self):
    return str(self)

  def _place_board_piece(self, row, col):
    piece_to_place = self._current_turn
    if self._board_state[row][col] != PIECE_EMPTY or row not in (0, 1, 2) or col not in (0, 1, 2):
      raise Exception(
          "Illegal action attempted by %(piece_to_place)s: (%(row)i, %(col)i)\nCurrent board:\n%(self)s" % locals())

    self._board_state[row][col] = piece_to_place

  def _player_has_won(self, player):
    board = self._board_state
    return (
      board[0][0] == player and board[0][1] == player and board[0][2] == player or
      board[1][0] == player and board[1][1] == player and board[1][2] == player or
      board[2][0] == player and board[2][1] == player and board[2][2] == player or

      board[0][0] == player and board[1][0] == player and board[2][0] == player or
      board[0][1] == player and board[1][1] == player and board[2][1] == player or
      board[0][2] == player and board[1][2] == player and board[2][2] == player or

      board[0][0] == player and board[1][1] == player and board[2][2] == player or
      board[0][2] == player and board[1][1] == player and board[2][0] == player
    )

  def _stringify_board_state(self):
    pretty_string = ""
    for row in self._board_state:
      if len(pretty_string) > 0:
        pretty_string += "\n"

      for piece in row:
        pretty_string += piece

    return pretty_string

  def __str__(self):
    return self._stringify_board_state()


class Agent:

  def __init__(self, epsilon = 1E-2, step_size = 1E-3, play_token = PIECE_X):
    self._epsilon = epsilon
    self._step_size = step_size
    self._play_token = play_token

    self._opponent_token = self._infer_opponent_token()

    self._value_table = {}
    self._board_states_observed = []

  def set_epsilon(self, epsilon):
    self._epsilon = epsilon

  def set_step_size(self, step_size):
    self._step_size = step_size

  def set_play_token(self, play_token):
    self._play_token = play_token
    self._opponent_token = self._infer_opponent_token()

  def get_greedy_action(self, board):
    return self._estimate_optimal_action(board)

  def get_training_action(self, board):
    if random.random() < self._epsilon:
      return self._randomly_choose_action(board)

    return self._estimate_optimal_action(board)

  def report_state_change(self, board):
    relative_board_state = self._relativize_board_state(board.get_immutable_board_state())
    self._board_states_observed.append(relative_board_state)

  def apply_reward(self, reward):
    final_game_state = self._board_states_observed.pop()
    self._value_table[final_game_state] = reward

    self._reinforce_actions_taken(final_game_state, self._board_states_observed)
    return None

  def clear_board_states(self):
    self._board_states_observed = []

  def get_value_table_values(self):
    return self._value_table.values()

  def _randomly_choose_action(self, board):
    available_actions = board.get_available_actions()
    return available_actions[random.randrange(0, len(available_actions))]

  def _estimate_optimal_action(self, board):
    available_actions = board.get_available_actions()

    estimated_optimal_action = available_actions.pop()
    action_consequence = board.mock_next_board_state(estimated_optimal_action)
    max_action_value_estimate = self._estimate_state_value(action_consequence)

    for action in available_actions:
      action_consequence = board.mock_next_board_state(action)
      value_estimate = self._estimate_state_value(action_consequence)

      if value_estimate > max_action_value_estimate:
        max_action_value_estimate = value_estimate
        estimated_optimal_action = action

    return estimated_optimal_action

  def _estimate_state_value(self, board_state):
    relative_board_state = self._relativize_board_state(board_state)
    if relative_board_state not in self._value_table.keys():
      self._value_table[relative_board_state] = 0.5

    return self._value_table[relative_board_state]

  def _reinforce_actions_taken(self, reinforced_board_state, remaining_states_to_reinforce):
    if len(remaining_states_to_reinforce) == 0:
      return

    preceding_board_state = remaining_states_to_reinforce.pop()

    board_state_value = self._estimate_state_value(reinforced_board_state)
    preceding_board_state_value = self._estimate_state_value(preceding_board_state)

    value_update_delta = self._step_size * (board_state_value - preceding_board_state_value)
    self._value_table[preceding_board_state] = preceding_board_state_value + value_update_delta

    self._reinforce_actions_taken(preceding_board_state, remaining_states_to_reinforce)

  def _relativize_board_state(self, board_state):
    own_tokens_replaced = board_state.replace(self._play_token, "P")
    all_tokens_replaced = own_tokens_replaced.replace(self._opponent_token, "Q")
    return all_tokens_replaced

  def _infer_opponent_token(self):
    pieces = [PIECE_X, PIECE_O]
    pieces.remove(self._play_token)
    return pieces[0]


def descriptive_stats(values):
  stats = {
      "max": -math.inf,
      "min": math.inf,
      "sum": 0,
  }

  for value in values:
    stats["sum"] += value
    if value > stats["max"]:
      stats["max"] = value
    if value < stats["min"]:
      stats["min"] = value

  stats["mean"] = stats["sum"] / float(len(values))

  return stats

def histogram(values, lower_bound=0.0, upper_bound=1.0, bucket_size=0.05):
  bucket_count = int((upper_bound - lower_bound) / bucket_size)
  counts = [0] * bucket_count

  for value in values:
    fraction = (value - lower_bound) / (upper_bound - lower_bound)
    index = int(fraction * bucket_count)

    index = max(index, 0)
    index = min(index, bucket_count - 1)

    counts[index] += 1

  return counts

def transpose(list_of_lists):
  return list(map(list, zip(*list_of_lists)))


def play_session(player_x, player_o, training_mode=True):
  board = XOBoard()

  player_x.report_state_change(board)
  player_o.report_state_change(board)

  current_player = player_x
  waiting_player = player_o

  while board.get_game_lifecycle_state() == STATE_NOT_FINISHED:
    action = None
    if training_mode:
      action = current_player.get_training_action(board)
    else:
      action = current_player.get_greedy_action(board)

    action_row, action_col = action
    board.place(action_row, action_col)

    current_player.report_state_change(board)
    waiting_player.report_state_change(board)

    tmp_player = current_player
    current_player = waiting_player
    waiting_player = tmp_player

  return board.get_game_lifecycle_state()


def play_greedy_session(agent_a, agent_b, randomized_first_player=False):
  player_x = agent_a
  player_o = agent_b
  if randomized_first_player and random.random() < 0.5:
    player_x = agent_b
    player_o = agent_a

  greedy_outcome = play_session(
    player_x,
    player_o,
    training_mode=False
  )
  agent_a.clear_board_states()
  agent_b.clear_board_states()

  return greedy_outcome

def play_training_sessions(agent_a, agent_b, number_sessions, randomized_first_player = False):
  training_game_outcomes = {
    STATE_WIN_X: 0,
    STATE_WIN_O: 0,
    STATE_DRAW: 0,
  }

  for _ in range(0, number_sessions):
    player_x = agent_a
    player_o = agent_b
    if randomized_first_player and random.random() < 0.5:
      player_x = agent_b
      player_o = agent_a

    game_result = play_session(
      player_x,
      player_o,
      training_mode=True,
    )

    training_game_outcomes[game_result] += 1

    x_reward = None
    o_reward = None
    if game_result == STATE_WIN_X:
      x_reward = 1.0
      o_reward = 0.0
    elif game_result == STATE_WIN_O:
      x_reward = 0.0
      o_reward = 1.0
    elif game_result == STATE_DRAW:
      x_reward = 0.0
      o_reward = 0.0
    else:
      raise Exception("play_session returned an unfinished game result.")

    player_x.apply_reward(x_reward)
    player_o.apply_reward(o_reward)
    player_x.clear_board_states()
    player_o.clear_board_states()

  return training_game_outcomes


def plot_training_regime(
    games_per_training_cycle=500,
    training_cycles=200,
    agent_a_epsilon=0.01,
    agent_b_epsilon=0.01,
    agent_a_step_size=0.01,
    agent_b_step_size=0.01,
    randomized_first_player=False,
  ):

  agent_a = Agent(agent_a_epsilon, agent_a_step_size)
  agent_b = Agent(agent_b_epsilon, agent_b_step_size)

  games_played = []
  x_victories = []
  o_victories = []
  draws = []

  x_value_histograms = []
  o_value_histograms = []
  for training_cycle in range(0, training_cycles):
    outcomes = play_training_sessions(
        agent_a,
        agent_b,
        games_per_training_cycle,
        randomized_first_player=randomized_first_player,
      )

    games_played.append(training_cycle)
    x_victories.append(outcomes[STATE_WIN_X] / games_per_training_cycle)
    o_victories.append(outcomes[STATE_WIN_O] / games_per_training_cycle)
    draws.append(outcomes[STATE_DRAW] / games_per_training_cycle)

    x_value_histograms.append(histogram(agent_a.get_value_table_values()))
    o_value_histograms.append(histogram(agent_b.get_value_table_values()))

  fig, ((ax, ax2, ax3)) = plt.subplots(3, 1, figsize=(15, 8), sharex=True)

  total_games_played = games_per_training_cycle * training_cycles
  plot_title_template = \
    "%(total_games_played)s Games Played - " + \
    "Agent A (ε = %(agent_a_epsilon)s, α = %(agent_a_step_size)s) vs. " + \
    "Agent B (ε = %(agent_b_epsilon)s, α = %(agent_b_step_size)s)\n " + \
    "X always goes first.  "
  if randomized_first_player:
    plot_title_template += "X is played by either agent."
  else:
    plot_title_template += "X is played only by Agent A."

  plot_title = plot_title_template % locals()
  plt.suptitle(plot_title)

  x_axis_label = "%(games_per_training_cycle)is of Games Played" % locals()
  ax.set(ylabel="Outcome Rate")

  ax.plot(games_played, x_victories, "-b", label=STATE_WIN_X)
  ax.plot(games_played, o_victories, "-g", label=STATE_WIN_O)
  ax.plot(games_played, draws, "-r", label=STATE_DRAW)
  ax.legend()

  x_value_heatmap = transpose(x_value_histograms)
  ax2.imshow(x_value_heatmap, aspect="auto", cmap="Reds")
  ax2.set(ylabel="Agent A - Value Distribution")
  ax2.yaxis.set_major_locator(ticker.MultipleLocator(base=2))
  ax2.set_yticklabels([""] + ["%0.1f" % (value * 0.01) for value in range(0, 110, 10)])

  o_value_heatmap = transpose(o_value_histograms)
  ax3.imshow(o_value_heatmap, aspect="auto", cmap="Reds")
  ax3.set(xlabel=x_axis_label, ylabel="Agent B - Value Distribution")
  ax3.yaxis.set_major_locator(ticker.MultipleLocator(base=2))
  ax3.set_yticklabels([""] + ["%0.1f" % (value * 0.01) for value in range(0, 110, 10)])

  foldername = "N=%(total_games_played)s R=%(randomized_first_player)s" % locals()
  os.makedirs(foldername, exist_ok=True)
  filename = "A(ε=%(agent_a_epsilon)s, α=%(agent_a_step_size)s) vs B(ε=%(agent_b_epsilon)s, α=%(agent_b_step_size)s).png" % locals()
  fig.savefig(os.path.join(foldername, filename))


if __name__ == "__main__":

  hyperparameter_template_string = \
    "games_per_training_cycle = %(games_per_training_cycle)s\n" + \
    "training_cycles = %(training_cycles)s\n" + \
    "agent_a_epsilon = %(agent_a_epsilon)s\n" + \
    "agent_b_epsilon = %(agent_b_epsilon)s\n" + \
    "agent_a_step_size = %(agent_a_step_size)s\n" + \
    "agent_b_step_size = %(agent_b_step_size)s\n" + \
    "randomized_first_player = %(randomized_first_player)s"

  training_cycles = 200
  for games_per_training_cycle in [1000, 10000]:
    for agent_a_epsilon in [0.1, 0.01]:
      for agent_b_epsilon in [0.1, 0.01]:
        for agent_a_step_size in [0.1, 0.01]:
          for agent_b_step_size in [0.1, 0.01]:
            for randomized_first_player in [True, False]:
              print("Training with hyperparameter set:")
              print(hyperparameter_template_string % locals())
              plot_training_regime(
                games_per_training_cycle=games_per_training_cycle,
                training_cycles=training_cycles,
                agent_a_epsilon=agent_a_epsilon,
                agent_b_epsilon=agent_b_epsilon,
                agent_a_step_size=agent_a_step_size,
                agent_b_step_size=agent_b_step_size,
                randomized_first_player=randomized_first_player,
              )
              print("Done.\n")

