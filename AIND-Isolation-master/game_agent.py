"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

CENTER_SQUARES = [(x, y) for x in [2, 3, 4] for y in [2, 3, 4]]
def center_moves(game, player):
    """Reward player's available moves in the center versus those of the
    opponent
    """
    opponent                 = game.get_opponent(player)
    player_moves             = game.get_legal_moves(player)
    opponent_moves           = game.get_legal_moves(opponent)
    player_center_moves      = float(len([m for m in player_moves   if m in CENTER_SQUARES]))
    opponent_center_moves    = float(len([m for m in opponent_moves if m in CENTER_SQUARES]))
    return player_center_moves - opponent_center_moves

def center_with_blank_moves(game, player):
    """
    Difference of player's center moves and opponent_center_moves 
    scaled by remaining blank spaces
    """
    blank_moves = float(len(game.get_blank_spaces()))
    return center_moves(game, player) / blank_moves

def moves_with_centers_and_blanks(game, player):
    """
    Calculates ratio of 
    player moves + center moves minus
    opponent moves + center moves - blank spaces 
    to the number of
    blank spaces + player moves - the opponent moves
    """
    opponent              = game.get_opponent(player)
    opponent_moves        = game.get_legal_moves(opponent)
    player_moves          = game.get_legal_moves(player)
    num_opponent_moves    = float(len(opponent_moves))
    num_player_moves      = float(len(player_moves))

    blank_moves           = float(len(game.get_blank_spaces()))
    player_center_moves   = float(len([m for m in player_moves   if m in CENTER_SQUARES]))
    opponent_center_moves = float(len([m for m in opponent_moves if m in CENTER_SQUARES]))

    return ((num_player_moves + player_center_moves) - (num_opponent_moves + opponent_center_moves) - blank_moves) \
           / (1 + blank_moves + num_player_moves - num_opponent_moves)

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    This should be the best heuristic function for your project submission.
    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")
    '''
    opponent           = game.get_opponent(player)
    opponent_moves     = game.get_legal_moves(opponent)
    player_moves       = game.get_legal_moves(player)
    num_opponent_moves = float(len(opponent_moves))
    num_player_moves   = float(len(player_moves))
    #''
    if num_opponent_moves == 0:
        return float("-inf")
    if num_player_moves == 0:
        return float("inf")
    '''
    # return (num_player_moves/num_opponent_moves) * (num_player_moves - num_opponent_moves)
    # return 2.0*(num_player_moves - num_opponent_moves)
    return moves_with_centers_and_blanks(game, player)




def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")
    opponent           = game.get_opponent(player)
    opponent_moves     = game.get_legal_moves(opponent)
    player_moves       = game.get_legal_moves(player)
    num_opponent_moves = float(len(opponent_moves))
    num_player_moves   = float(len(player_moves))


    return 4.0*num_player_moves - num_opponent_moves


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")
    opponent           = game.get_opponent(player)
    opponent_moves     = game.get_legal_moves(opponent)
    player_moves       = game.get_legal_moves(player)
    num_opponent_moves = float(len(opponent_moves))
    num_player_moves   = float(len(player_moves))

    return num_player_moves - 4.0*num_opponent_moves


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.
    ********************  DO NOT MODIFY THIS CLASS  ********************
    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)
    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.
    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.
        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************
        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.
        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.
        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return (-1, -1)
        #center = (game.width / 2, game.height / 2)
        #best_move = center if center in legal_moves else random.choice(legal_moves)
        center_legal_moves = [move for move in legal_moves if move in CENTER_SQUARES]
        best_move = random.choice(center_legal_moves) if center_legal_moves else random.choice(legal_moves)
        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            return best_move   # Handle any actions required after timeout as needed

        finally:
            return best_move

    def minval(self, game, depth):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()

        # Final depth terminal test
        if depth == 0 or (not legal_moves):
            return self.score(game, self)

        best_score = float('inf')
        for current_move in legal_moves:
            forecast_game = game.forecast_move(current_move)
            score = self.maxval(forecast_game, depth - 1)
            if score < best_score:
                best_score = score
        return best_score

    def maxval(self, game, depth):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()
        # Final depth terminal test
        if depth == 0 or (not legal_moves):
            return self.score(game, self)

        best_score = -float('inf')
        for current_move in legal_moves:
            forecast_game = game.forecast_move(current_move)
            score = self.minval(forecast_game, depth - 1)
            if score > best_score:
                best_score = score
        return best_score

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.
        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md
        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.
            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!
        best_score, best_move, = -float('inf'), None
        legal_moves = game.get_legal_moves()
        for current_move in legal_moves:
            forecast_game = game.forecast_move(current_move)
            score = self.minval(forecast_game, depth - 1)
            if score > best_score:
                best_score, best_move = score, current_move
        return best_move


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.
        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.
        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************
        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.
        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # TODO: finish this function!
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return (-1, -1)
        #center = (game.width / 2, game.height / 2)
        center_legal_moves = [move for move in legal_moves if move in CENTER_SQUARES]
        best_move = random.choice(center_legal_moves) if center_legal_moves else random.choice(legal_moves)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            temp_depth = 1
            while (self.time_left() > 0):
                best_move = self.alphabeta(game, temp_depth)
                temp_depth += 1

        except SearchTimeout:
            return best_move   # Handle any actions required after timeout as needed

        finally:
            return best_move

    def minval_alphabeta(self, game, depth, alpha, beta):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        legal_moves = game.get_legal_moves()

        # Final depth terminal test
        if depth == 0 or (not legal_moves):
            return self.score(game, self)

        best_score = float('inf')
        for current_move in legal_moves:
            forecast_game = game.forecast_move(current_move)
            score = self.maxval_alphabeta(forecast_game, depth - 1, alpha, beta)
            best_score = min(best_score,score)
            if best_score <= alpha:     return best_score
            beta = min(beta, best_score)

        return best_score

    def maxval_alphabeta(self, game, depth, alpha, beta):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()

        # Final depth terminal test
        if depth == 0 or (not legal_moves):
            return self.score(game, self)

        best_score = -float('inf')
        for current_move in legal_moves:
            forecast_game = game.forecast_move(current_move)
            score = self.minval_alphabeta(forecast_game, depth - 1, alpha, beta)
            best_score = max(best_score,score)
            if best_score >= beta:      return best_score
            alpha = max(alpha, best_score)

        return best_score

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.
        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md
        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        alpha : float
            Alpha limits the lower bound of search on minimizing layers
        beta : float
            Beta limits the upper bound of search on maximizing layers
        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.
            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!
        best_score, best_move, = -float('inf'), None
        legal_moves = game.get_legal_moves()
        for current_move in legal_moves:
            forecast_game = game.forecast_move(current_move)
            score = self.minval_alphabeta(forecast_game, depth - 1, alpha, beta)
            if score >= best_score:
                best_score, best_move = score, current_move
            if best_score >= beta:      return best_move
            alpha = max(alpha, score)

        return best_move


