/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2018 Gian-Carlo Pascutto and contributors

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
*/

#ifndef TIMECONTROL_H_INCLUDED
#define TIMECONTROL_H_INCLUDED

#include <array>

#include "config.h"
#include "Timing.h"

class TimeControl {
public:
    /*
        Initialize time control. Timing info is per GTP and in centiseconds
    */
<<<<<<< HEAD
    TimeControl(int boardsize = BOARD_SIZE,
                int maintime = 60 * 60 * 100,
=======
    TimeControl(int maintime = 60 * 60 * 100,
>>>>>>> upstream/master
                int byotime = 0, int byostones = 25,
                int byoperiods = 0);

    void start(int color);
    void stop(int color);
<<<<<<< HEAD
    int max_time_for_move(int color, int movenum);
=======
    int max_time_for_move(int boardsize, int color, size_t movenum) const;
>>>>>>> upstream/master
    void adjust_time(int color, int time, int stones);
    void display_times();
    void reset_clocks();
<<<<<<< HEAD
    bool can_accumulate_time(int color);
    std::string to_text_sgf();

private:
    void display_color_time(int color);
    int get_moves_expected(int movenum);
=======
    bool can_accumulate_time(int color) const;
    size_t opening_moves(int boardsize) const;
    std::string to_text_sgf() const;

private:
    void display_color_time(int color);
    int get_moves_expected(int boardsize, size_t movenum) const;
>>>>>>> upstream/master

    int m_maintime;
    int m_byotime;
    int m_byostones;
    int m_byoperiods;
<<<<<<< HEAD
    int m_boardsize;
=======
>>>>>>> upstream/master

    std::array<int,  2> m_remaining_time;    /* main time per player */
    std::array<int,  2> m_stones_left;       /* stones to play in byo period */
    std::array<int,  2> m_periods_left;      /* byo periods */
    std::array<bool, 2> m_inbyo;             /* player is in byo yomi */

    std::array<Time, 2> m_times;             /* storage for player times */
};

#endif
