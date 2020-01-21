/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with NVIDIA Corporation's libraries from the
    NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
    Network library and/or the NVIDIA TensorRT inference library
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective license agreement, the licensors of
    this Program grant you additional permission to convey the resulting
    work.
*/

#include "GameState.h"
#include "Network.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <iterator>
#include <memory>
#include <string>

#include "FastBoard.h"
#include "FastState.h"
#include "FullBoard.h"
#include "KoState.h"
#include "UCTSearch.h"

#include "Utils.h"
using namespace Utils;

void GameState::init_game(int size, float komi) {
    KoState::init_game(size, komi);

    game_history.clear();
    game_history.emplace_back(std::make_shared<KoState>(*this));

    m_timecontrol.reset_clocks();

    m_resigned = FastBoard::EMPTY;
}

void GameState::reset_game() {
    KoState::reset_game();

    game_history.clear();
    game_history.emplace_back(std::make_shared<KoState>(*this));

    m_timecontrol.reset_clocks();

    m_resigned = FastBoard::EMPTY;
}

bool GameState::forward_move() {
    if (game_history.size() > m_movenum + 1) {
        m_movenum++;
        *(static_cast<KoState*>(this)) = *game_history[m_movenum];
        return true;
    } else {
        return false;
    }
}

bool GameState::undo_move() {
    if (m_movenum > 0) {
        m_movenum--;

        // this is not so nice, but it should work
        *(static_cast<KoState*>(this)) = *game_history[m_movenum];

        // This also restores hashes as they're part of state
        return true;
    } else {
        return false;
    }
}

void GameState::rewind() {
    *(static_cast<KoState*>(this)) = *game_history[0];
    m_movenum = 0;
}

void GameState::play_move(int vertex) {
    play_move(get_to_move(), vertex);
}

void GameState::play_move(int color, int vertex) {
    if (vertex == FastBoard::RESIGN) {
        m_resigned = color;
    } else {
        KoState::play_move(color, vertex);
    }

    // cut off any leftover moves from navigating
    game_history.resize(m_movenum);
    game_history.emplace_back(std::make_shared<KoState>(*this));
}

bool GameState::play_textmove(std::string color, const std::string& vertex) {
    int who;
    transform(cbegin(color), cend(color), begin(color), tolower);
    if (color == "w" || color == "white") {
        who = FullBoard::WHITE;
    } else if (color == "b" || color == "black") {
        who = FullBoard::BLACK;
    } else {
        return false;
    }

    const auto move = board.text_to_move(vertex);
    if (move == FastBoard::NO_VERTEX ||
        (move != FastBoard::PASS && move != FastBoard::RESIGN && board.get_state(move) != FastBoard::EMPTY)) {
        return false;
    }

    set_to_move(who);
    play_move(move);

    return true;
}

void GameState::stop_clock(int color) {
    m_timecontrol.stop(color);
}

void GameState::start_clock(int color) {
    m_timecontrol.start(color);
}

void GameState::display_state() {
    FastState::display_state();

    m_timecontrol.display_times();
}

int GameState::who_resigned() const {
    return m_resigned;
}

bool GameState::has_resigned() const {
    return m_resigned != FastBoard::EMPTY;
}

const TimeControl& GameState::get_timecontrol() const {
    return m_timecontrol;
}

void GameState::set_timecontrol(const TimeControl& timecontrol) {
    m_timecontrol = timecontrol;
}

void GameState::set_timecontrol(int maintime, int byotime,
                                int byostones, int byoperiods) {
    TimeControl timecontrol(maintime, byotime,
                            byostones, byoperiods);

    m_timecontrol = timecontrol;
}

void GameState::adjust_time(int color, int time, int stones) {
    m_timecontrol.adjust_time(color, time, stones);
}

void GameState::anchor_game_history() {
    // handicap moves don't count in game history
    m_movenum = 0;
    game_history.clear();
    game_history.emplace_back(std::make_shared<KoState>(*this));
}

bool GameState::set_fixed_handicap(int handicap) {
    if (!valid_handicap(handicap)) {
        return false;
    }

    int board_size = board.get_boardsize();
    int high = board_size >= 13 ? 3 : 2;
    int mid = board_size / 2;

    int low = board_size - 1 - high;
    if (handicap >= 2) {
        play_move(FastBoard::BLACK, board.get_vertex(low, low));
        play_move(FastBoard::BLACK, board.get_vertex(high, high));
    }

    if (handicap >= 3) {
        play_move(FastBoard::BLACK, board.get_vertex(high, low));
    }

    if (handicap >= 4) {
        play_move(FastBoard::BLACK, board.get_vertex(low, high));
    }

    if (handicap >= 5 && handicap % 2 == 1) {
        play_move(FastBoard::BLACK, board.get_vertex(mid, mid));
    }

    if (handicap >= 6) {
        play_move(FastBoard::BLACK, board.get_vertex(low, mid));
        play_move(FastBoard::BLACK, board.get_vertex(high, mid));
    }

    if (handicap >= 8) {
        play_move(FastBoard::BLACK, board.get_vertex(mid, low));
        play_move(FastBoard::BLACK, board.get_vertex(mid, high));
    }

    board.set_to_move(FastBoard::WHITE);

    anchor_game_history();

    set_handicap(handicap);

    return true;
}

int GameState::set_fixed_handicap_2(int handicap) {
    int board_size = board.get_boardsize();
    int low = board_size >= 13 ? 3 : 2;
    int mid = board_size / 2;
    int high = board_size - 1 - low;

    int interval = (high - mid) / 2;
    int placed = 0;

    while (interval >= 3) {
        for (int i = low; i <= high; i += interval) {
            for (int j = low; j <= high; j += interval) {
                if (placed >= handicap) return placed;
                if (board.get_state(i-1, j-1) != FastBoard::EMPTY) continue;
                if (board.get_state(i-1, j) != FastBoard::EMPTY) continue;
                if (board.get_state(i-1, j+1) != FastBoard::EMPTY) continue;
                if (board.get_state(i, j-1) != FastBoard::EMPTY) continue;
                if (board.get_state(i, j) != FastBoard::EMPTY) continue;
                if (board.get_state(i, j+1) != FastBoard::EMPTY) continue;
                if (board.get_state(i+1, j-1) != FastBoard::EMPTY) continue;
                if (board.get_state(i+1, j) != FastBoard::EMPTY) continue;
                if (board.get_state(i+1, j+1) != FastBoard::EMPTY) continue;
                play_move(FastBoard::BLACK, board.get_vertex(i, j));
                placed++;
            }
        }
        interval = interval / 2;
    }

    return placed;
}

bool GameState::valid_handicap(int handicap) {
    int board_size = board.get_boardsize();

    if (handicap < 2 || handicap > 9) {
        return false;
    }
    if (board_size % 2 == 0 && handicap > 4) {
        return false;
    }
    if (board_size == 7 && handicap > 4) {
        return false;
    }
    if (board_size < 7 && handicap > 0) {
        return false;
    }

    return true;
}

void GameState::place_free_handicap(int stones, Network & network, Network & network_aux) {
    int limit = board.get_boardsize() * board.get_boardsize();
    if (stones > limit / 2) {
        stones = limit / 2;
    }

    int orgstones = stones;

    int fixplace = std::min(9, stones);

    set_fixed_handicap(fixplace);
    stones -= fixplace;

    stones -= set_fixed_handicap_2(stones);

    for (int i = 0; i < stones; i++) {
        auto search = std::make_unique<UCTSearch>(*this, network, network_aux);
        auto move = search->think(FastBoard::BLACK, UCTSearch::NOPASS);
        play_move(FastBoard::BLACK, move);
    }

    if (orgstones)  {
        board.set_to_move(FastBoard::WHITE);
    } else {
        board.set_to_move(FastBoard::BLACK);
    }

    anchor_game_history();

    set_handicap(orgstones);
}

const FullBoard& GameState::get_past_board(int moves_ago) const {
    assert(moves_ago >= 0 && (unsigned)moves_ago <= m_movenum);
    assert(m_movenum + 1 <= game_history.size());
    return game_history[m_movenum - moves_ago]->board;
}

const std::vector<std::shared_ptr<const KoState>>& GameState::get_game_history() const {
    return game_history;
}

int GameState::get_move(int n) const {
    if (n>m_movenum) {
        return FastBoard::NO_VERTEX;
    }
    for (const auto &s : game_history) {
        auto num = s->get_movenum();
        if (num<n) { continue; }
        return s->get_last_move();
    }
    return FastBoard::NO_VERTEX;
}

Ladder::LadderStatus Ladder::ladder_status(const FastState &state) {
    const auto board = state.board;

    Ladder::LadderStatus status;

    for (auto i = 0; i < BOARD_SIZE; i++) {
        for (auto j = 0; j < BOARD_SIZE; j++) {
            auto vertex = board.get_vertex(i, j);
            status[i][j] = NO_LADDER;
            if (ladder_capture(state, vertex)) {
                status[i][j] = CAPTURE;
            }
            if (ladder_escape(state, vertex)) {
                status[i][j] = ESCAPE;
            }
        }
    }
    return status;
}

bool Ladder::ladder_capture(const FastState &state, int vertex, int group, int depth) {

    const auto &board = state.board;
    const auto capture_player = board.get_to_move();
    const auto escape_player = capture_player==FastBoard::WHITE?FastBoard::BLACK:FastBoard::WHITE;

    if (!state.is_move_legal(capture_player, vertex)) {
        return false;
    }

    // Assume that capture succeeds if it takes this long
    if (depth >= 100) {
        return true;
    }

    std::vector<int> groups_in_ladder;

    if (group == FastBoard::PASS) {
        // Check if there are nearby groups with 2 liberties
        for (int d = 0; d < 4; d++) {
            int n_vtx = board.get_neighbor(vertex, d);
            int n = board.get_state(n_vtx);
            if ((n == escape_player) && (board.get_lib(n_vtx) == 2)) {
                auto parent = board.get_parent(n_vtx);
                if (std::find(groups_in_ladder.begin(), groups_in_ladder.end(), parent) == groups_in_ladder.end()) {
                    groups_in_ladder.emplace_back(parent);
                }
            }
        }
    } else {
        groups_in_ladder.emplace_back(group);
    }

    for (auto& group : groups_in_ladder) {
        auto state_copy = std::make_unique<FastState>(state);
        auto &board_copy = state_copy->board;

        state_copy->play_move(vertex);

        int escape = FastBoard::PASS;
        int newpos = group;
        do {
            for (int d = 0; d < 4; d++) {
                int stone = board_copy.get_neighbor(newpos, d);
                // If the surrounding stones are in atari capture fails
                if (board_copy.get_state(stone) == capture_player) {
                    if (board_copy.get_lib(stone) == 1) {
                        return false;
                    }
                }
                // Possible move to escape
                if (board_copy.get_state(stone) == FastBoard::EMPTY) {
                    escape = stone;
                }
            }
            newpos = board_copy.get_next(newpos);
        } while (newpos != group);
        
        assert(escape != FastBoard::PASS);

        // If escaping fails the capture was successful
        if (!ladder_escape(*state_copy, escape, group, depth + 1)) {
            return true;
        }
    }

    return false;
}

bool Ladder::ladder_escape(const FastState &state, const int vertex, int group, int depth) {
    const auto &board = state.board;
    const auto escape_player = board.get_to_move();

    if (!state.is_move_legal(escape_player, vertex)) {
        return false;
    }

    // Assume that escaping failed if it takes this long
    if (depth >= 100) {
        return false;
    }

    std::vector<int> groups_in_ladder;

    if (group == FastBoard::PASS) {
        // Check if there are nearby groups with 1 liberties
        for (int d = 0; d < 4; d++) {
            int n_vtx = board.get_neighbor(vertex, d);
            int n = board.get_state(n_vtx);
            if ((n == escape_player) && (board.get_lib(n_vtx) == 1)) {
                auto parent = board.get_parent(n_vtx);
                if (std::find(groups_in_ladder.begin(), groups_in_ladder.end(), parent) == groups_in_ladder.end()) {
                    groups_in_ladder.emplace_back(parent);
                }
            }
        }
    } else {
        groups_in_ladder.emplace_back(group);
    }

    for (auto& group : groups_in_ladder) {
        auto state_copy = std::make_unique<FastState>(state);
        auto &board_copy = state_copy->board;

        state_copy->play_move(vertex);

        if (board_copy.get_lib(group) >= 3) {
            // Opponent can't atari on the next turn
            return true;
        }

        if (board_copy.get_lib(group) == 1) {
            // Will get captured on the next turn
            return false;
        }

        // Still two liberties left, check for possible captures
        int newpos = group;
        do {
            for (int d = 0; d < 4; d++) {
                int empty = board_copy.get_neighbor(newpos, d);
                if (board_copy.get_state(empty) == FastBoard::EMPTY) {
                    if (ladder_capture(*state_copy, empty, group, depth + 1)) {
                        // Got captured
                        return false;
                    }
                }
            }
            newpos = board_copy.get_next(newpos);
        } while (newpos != group);

        // Ladder capture failed, escape succeeded
        return true;
    }

    return false;
}

static void print_columns() {
    for (int i = 0; i < BOARD_SIZE; i++) {
        if (i < 25) {
            myprintf("%c ", (('a' + i < 'i') ? 'a' + i : 'a' + i + 1));
        }
        else {
            myprintf("%c ", (('A' + (i - 25) < 'I') ? 'A' + (i - 25) : 'A' + (i - 25) + 1));
        }
    }
    myprintf("\n");
}

void Ladder::display_ladders(const LadderStatus &status) {
    myprintf("\n   ");
    print_columns();
    for (int j = BOARD_SIZE-1; j >= 0; j--) {
        myprintf("%2d", j+1);
        myprintf(" ");
        for (int i = 0; i < BOARD_SIZE; i++) {
            if (status[i][j] == CAPTURE) {
                myprintf("C");
            } else if (status[i][j] == ESCAPE) {
                myprintf("E");
            } else if (FastBoard::starpoint(BOARD_SIZE, i, j)) {
                myprintf("+");
            } else {
                myprintf(".");
            }
            myprintf(" ");
        }
        myprintf("%2d\n", j+1);
    }
    myprintf("   ");
    print_columns();
    myprintf("\n");
}

void Ladder::display_ladders(const FastState &state) {
    display_ladders(ladder_status(state));
}
