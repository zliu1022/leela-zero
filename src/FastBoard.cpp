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

#include "FastBoard.h"

#include <cassert>
#include <cctype>
#include <algorithm>
#include <array>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>

#include "Utils.h"
#include "config.h"

using namespace Utils;

const int FastBoard::NBR_SHIFT;
const int FastBoard::NUM_VERTICES;
const int FastBoard::NO_VERTEX;
const int FastBoard::PASS;
const int FastBoard::RESIGN;

const std::array<int, 2> FastBoard::s_eyemask = {
    4 * (1 << (NBR_SHIFT * BLACK)),
    4 * (1 << (NBR_SHIFT * WHITE))
};

const std::array<FastBoard::vertex_t, 4> FastBoard::s_cinvert = {
    WHITE, BLACK, EMPTY, INVAL
};

int FastBoard::get_boardsize() const {
    return m_boardsize;
}

int FastBoard::get_vertex(int x, int y) const {
    assert(x >= 0 && x < BOARD_SIZE);
    assert(y >= 0 && y < BOARD_SIZE);
    assert(x >= 0 && x < m_boardsize);
    assert(y >= 0 && y < m_boardsize);

    int vertex = ((y + 1) * m_sidevertices) + (x + 1);

    assert(vertex >= 0 && vertex < m_numvertices);

    return vertex;
}

std::pair<int, int> FastBoard::get_xy(int vertex) const {
    //int vertex = ((y + 1) * (get_boardsize() + 2)) + (x + 1);
    int x = (vertex % m_sidevertices) - 1;
    int y = (vertex / m_sidevertices) - 1;

    assert(x >= 0 && x < m_boardsize);
    assert(y >= 0 && y < m_boardsize);
    assert(get_vertex(x, y) == vertex);

    return std::make_pair(x, y);
}

FastBoard::vertex_t FastBoard::get_state(int vertex) const {
    assert(vertex >= 0 && vertex < NUM_VERTICES);
    assert(vertex >= 0 && vertex < m_numvertices);

    return m_state[vertex];
}

void FastBoard::set_state(int vertex, FastBoard::vertex_t content) {
    assert(vertex >= 0 && vertex < NUM_VERTICES);
    assert(vertex >= 0 && vertex < m_numvertices);
    assert(content >= BLACK && content <= INVAL);

    m_state[vertex] = content;
}

FastBoard::vertex_t FastBoard::get_state(int x, int y) const {
    return get_state(get_vertex(x, y));
}

void FastBoard::set_state(int x, int y, FastBoard::vertex_t content) {
    set_state(get_vertex(x, y), content);
}

void FastBoard::reset_board(int size) {
    m_boardsize = size;
    m_sidevertices = size + 2;
    m_numvertices = m_sidevertices * m_sidevertices;
    m_tomove = BLACK;
    m_prisoners[BLACK] = 0;
    m_prisoners[WHITE] = 0;
    m_empty_cnt = 0;

    m_dirs[0] = -m_sidevertices;
    m_dirs[1] = +1;
    m_dirs[2] = +m_sidevertices;
    m_dirs[3] = -1;

    for (int i = 0; i < m_numvertices; i++) {
        m_state[i]     = INVAL;
        m_neighbours[i] = 0;
        m_parent[i]     = NUM_VERTICES;
    }

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int vertex = get_vertex(i, j);

            m_state[vertex]           = EMPTY;
            m_empty_idx[vertex]       = m_empty_cnt;
            m_empty[m_empty_cnt++]    = vertex;

            if (i == 0 || i == size - 1) {
                m_neighbours[vertex] += (1 << (NBR_SHIFT * BLACK))
                                      | (1 << (NBR_SHIFT * WHITE));
                m_neighbours[vertex] +=  1 << (NBR_SHIFT * EMPTY);
            } else {
                m_neighbours[vertex] +=  2 << (NBR_SHIFT * EMPTY);
            }

            if (j == 0 || j == size - 1) {
                m_neighbours[vertex] += (1 << (NBR_SHIFT * BLACK))
                                      | (1 << (NBR_SHIFT * WHITE));
                m_neighbours[vertex] +=  1 << (NBR_SHIFT * EMPTY);
            } else {
                m_neighbours[vertex] +=  2 << (NBR_SHIFT * EMPTY);
            }
        }
    }

    m_parent[NUM_VERTICES] = NUM_VERTICES;
    m_libs[NUM_VERTICES]   = 16384;    /* we will subtract from this */
    m_next[NUM_VERTICES]   = NUM_VERTICES;

    assert(m_state[NO_VERTEX] == INVAL);
}

bool FastBoard::is_suicide(int i, int color) const {
    // If there are liberties next to us, it is never suicide
    if (count_pliberties(i)) {
        return false;
    }

    // If we get here, we played in a "hole" surrounded by stones
    for (auto k = 0; k < 4; k++) {
        auto ai = i + m_dirs[k];

        auto libs = m_libs[m_parent[ai]];
        if (get_state(ai) == color) {
            if (libs > 1) {
                // connecting to live group = not suicide
                return false;
            }
        } else if (get_state(ai) == !color) {
            if (libs <= 1) {
                // killing neighbour = not suicide
                return false;
            }
        }
    }

    // We played in a hole, friendlies had one liberty at most and
    // we did not kill anything. So we killed ourselves.
    return true;
}

int FastBoard::count_pliberties(const int i) const {
    return count_neighbours(EMPTY, i);
}

// count neighbours of color c at vertex v
// the border of the board has fake neighours of both colors
int FastBoard::count_neighbours(const int c, const int v) const {
    assert(c == WHITE || c == BLACK || c == EMPTY);
    return (m_neighbours[v] >> (NBR_SHIFT * c)) & NBR_MASK;
}

void FastBoard::add_neighbour(const int vtx, const int color) {
    assert(color == WHITE || color == BLACK || color == EMPTY);

    std::array<int, 4> nbr_pars;
    int nbr_par_cnt = 0;

    for (int k = 0; k < 4; k++) {
        int ai = vtx + m_dirs[k];

        m_neighbours[ai] += (1 << (NBR_SHIFT * color)) - (1 << (NBR_SHIFT * EMPTY));

        bool found = false;
        for (int i = 0; i < nbr_par_cnt; i++) {
            if (nbr_pars[i] == m_parent[ai]) {
                found = true;
                break;
            }
        }
        if (!found) {
            m_libs[m_parent[ai]]--;
            nbr_pars[nbr_par_cnt++] = m_parent[ai];
        }
    }
}

void FastBoard::remove_neighbour(const int vtx, const int color) {
    assert(color == WHITE || color == BLACK || color == EMPTY);

    std::array<int, 4> nbr_pars;
    int nbr_par_cnt = 0;

    for (int k = 0; k < 4; k++) {
        int ai = vtx + m_dirs[k];

        m_neighbours[ai] += (1 << (NBR_SHIFT * EMPTY))
                          - (1 << (NBR_SHIFT * color));

        bool found = false;
        for (int i = 0; i < nbr_par_cnt; i++) {
            if (nbr_pars[i] == m_parent[ai]) {
                found = true;
                break;
            }
        }
        if (!found) {
            m_libs[m_parent[ai]]++;
            nbr_pars[nbr_par_cnt++] = m_parent[ai];
        }
    }
}

int FastBoard::calc_reach_color(int color) const {
    auto reachable = 0;
    auto bd = std::vector<bool>(m_numvertices, false);
    auto open = std::queue<int>();
    for (auto i = 0; i < m_boardsize; i++) {
        for (auto j = 0; j < m_boardsize; j++) {
            auto vertex = get_vertex(i, j);
            if (m_state[vertex] == color) {
                reachable++;
                bd[vertex] = true;
                open.push(vertex);
            }
        }
    }
    while (!open.empty()) {
        /* colored field, spread */
        auto vertex = open.front();
        open.pop();

        for (auto k = 0; k < 4; k++) {
            auto neighbor = vertex + m_dirs[k];
            if (!bd[neighbor] && m_state[neighbor] == EMPTY) {
                reachable++;
                bd[neighbor] = true;
                open.push(neighbor);
            }
        }
    }
    return reachable;
}

// Needed for scoring passed out games not in MC playouts
float FastBoard::area_score(float komi) const {
    auto white = calc_reach_color(WHITE);
    auto black = calc_reach_color(BLACK);
    return black - white - komi;
}

void FastBoard::display_board(int lastmove) {
    int boardsize = get_boardsize();

    myprintf("\n   ");
    print_columns();
    for (int j = boardsize-1; j >= 0; j--) {
        myprintf("%2d", j+1);
        if (lastmove == get_vertex(0, j))
            myprintf("(");
        else
            myprintf(" ");
        for (int i = 0; i < boardsize; i++) {
            if (get_state(i,j) == WHITE) {
                myprintf("O");
            } else if (get_state(i,j) == BLACK)  {
                myprintf("X");
            } else if (starpoint(boardsize, i, j)) {
                myprintf("+");
            } else {
                myprintf(".");
            }
            if (lastmove == get_vertex(i, j)) myprintf(")");
            else if (i != boardsize-1 && lastmove == get_vertex(i, j)+1) myprintf("(");
            else myprintf(" ");
        }
        myprintf("%2d\n", j+1);
    }
    myprintf("   ");
    print_columns();
    myprintf("\n");
}

void FastBoard::print_columns() {
    for (int i = 0; i < get_boardsize(); i++) {
        if (i < 25) {
            myprintf("%c ", (('a' + i < 'i') ? 'a' + i : 'a' + i + 1));
        } else {
            myprintf("%c ", (('A' + (i - 25) < 'I') ? 'A' + (i - 25) : 'A' + (i - 25) + 1));
        }
    }
    myprintf("\n");
}

void FastBoard::merge_strings(const int ip, const int aip) {
    assert(ip != NUM_VERTICES && aip != NUM_VERTICES);

    /* merge stones */
    m_stones[ip] += m_stones[aip];

    /* loop over stones, update parents */
    int newpos = aip;

    do {
        // check if this stone has a liberty
        for (int k = 0; k < 4; k++) {
            int ai = newpos + m_dirs[k];
            // for each liberty, check if it is not shared
            if (m_state[ai] == EMPTY) {
                // find liberty neighbors
                bool found = false;
                for (int kk = 0; kk < 4; kk++) {
                    int aai = ai + m_dirs[kk];
                    // friendly string shouldn't be ip
                    // ip can also be an aip that has been marked
                    if (m_parent[aai] == ip) {
                        found = true;
                        break;
                    }
                }

                if (!found) {
                    m_libs[ip]++;
                }
            }
        }

        m_parent[newpos] = ip;
        newpos = m_next[newpos];
    } while (newpos != aip);

    /* merge stings */
    std::swap(m_next[aip], m_next[ip]);
}

bool FastBoard::is_eye(const int color, const int i) const {
    /* check for 4 neighbors of the same color */
    int ownsurrounded = (m_neighbours[i] & s_eyemask[color]);

    // if not, it can't be an eye
    // this takes advantage of borders being colored
    // both ways
    if (!ownsurrounded) {
        return false;
    }

    // 2 or more diagonals taken
    // 1 for side groups
    int colorcount[4];

    colorcount[BLACK] = 0;
    colorcount[WHITE] = 0;
    colorcount[INVAL] = 0;

    colorcount[m_state[i - 1 - m_sidevertices]]++;
    colorcount[m_state[i + 1 - m_sidevertices]]++;
    colorcount[m_state[i - 1 + m_sidevertices]]++;
    colorcount[m_state[i + 1 + m_sidevertices]]++;

    if (colorcount[INVAL] == 0) {
        if (colorcount[!color] > 1) {
            return false;
        }
    } else {
        if (colorcount[!color]) {
            return false;
        }
    }

    return true;
}

std::string FastBoard::move_to_text(int move) const {
    std::ostringstream result;

    int column = move % m_sidevertices;
    int row = move / m_sidevertices;

    column--;
    row--;

    assert(move == FastBoard::PASS
           || move == FastBoard::RESIGN
           || (row >= 0 && row < m_boardsize));
    assert(move == FastBoard::PASS
           || move == FastBoard::RESIGN
           || (column >= 0 && column < m_boardsize));

    if (move >= 0 && move <= m_numvertices) {
        result << static_cast<char>(column < 8 ? 'A' + column : 'A' + column + 1);
        result << (row + 1);
    } else if (move == FastBoard::PASS) {
        result << "pass";
    } else if (move == FastBoard::RESIGN) {
        result << "resign";
    } else {
        result << "error";
    }

    return result.str();
}

int FastBoard::text_to_move(std::string move) const {
    transform(cbegin(move), cend(move), begin(move), tolower);

    if (move == "pass") {
        return PASS;
    } else if (move == "resign") {
        return RESIGN;
    } else if (move.size() < 2 || !std::isalpha(move[0]) || !std::isdigit(move[1]) || move[0] == 'i') {
        return NO_VERTEX;
    }

    auto column = move[0] - 'a';
    if (move[0] > 'i') {
        --column;
    }

    int row;
    std::istringstream parsestream(move.substr(1));
    parsestream >> row;
    --row;

    if (row >= m_boardsize || column >= m_boardsize) {
        return NO_VERTEX;
    }

    return get_vertex(column, row);
}

std::string FastBoard::move_to_text_sgf(int move) const {
    std::ostringstream result;

    int column = move % m_sidevertices;
    int row = move / m_sidevertices;

    column--;
    row--;

    assert(move == FastBoard::PASS
           || move == FastBoard::RESIGN
           || (row >= 0 && row < m_boardsize));
    assert(move == FastBoard::PASS
           || move == FastBoard::RESIGN
           || (column >= 0 && column < m_boardsize));

    // SGF inverts rows
    row = m_boardsize - row - 1;

    if (move >= 0 && move <= m_numvertices) {
        if (column <= 25) {
            result << static_cast<char>('a' + column);
        } else {
            result << static_cast<char>('A' + column - 26);
        }
        if (row <= 25) {
            result << static_cast<char>('a' + row);
        } else {
            result << static_cast<char>('A' + row - 26);
        }
    } else if (move == FastBoard::PASS) {
        result << "tt";
    } else if (move == FastBoard::RESIGN) {
        result << "tt";
    } else {
        result << "error";
    }

    return result.str();
}

bool FastBoard::starpoint(int size, int point) {
    int stars[3];
    int points[2];
    int hits = 0;

    if (size % 2 == 0 || size < 9) {
        return false;
    }

    stars[0] = size >= 13 ? 3 : 2;
    stars[1] = size / 2;
    stars[2] = size - 1 - stars[0];

    points[0] = point / size;
    points[1] = point % size;

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            if (points[i] == stars[j]) {
                hits++;
            }
        }
    }

    return hits >= 2;
}

bool FastBoard::starpoint(int size, int x, int y) {
    return starpoint(size, y * size + x);
}

int FastBoard::get_prisoners(int side)  const {
    assert(side == WHITE || side == BLACK);

    return m_prisoners[side];
}

int FastBoard::get_to_move() const {
    return m_tomove;
}

bool FastBoard::black_to_move() const {
    return m_tomove == BLACK;
}

bool FastBoard::white_to_move() const {
    return m_tomove == WHITE;
}

void FastBoard::set_to_move(int tomove) {
    m_tomove = tomove;
}

std::string FastBoard::get_string(int vertex) const {
    std::string result;

    int start = m_parent[vertex];
    int newpos = start;

    do {
        result += move_to_text(newpos) + " ";
        newpos = m_next[newpos];
    } while (newpos != start);

    // eat last space
    assert(result.size() > 0);
    result.resize(result.size() - 1);

    return result;
}

std::string FastBoard::get_stone_list() const {
    std::string result;

    for (int i = 0; i < m_boardsize; i++) {
        for (int j = 0; j < m_boardsize; j++) {
            int vertex = get_vertex(i, j);

            if (get_state(vertex) != EMPTY) {
                result += move_to_text(vertex) + " ";
            }
        }
    }

    // eat final space, if any.
    if (result.size() > 0) {
        result.resize(result.size() - 1);
    }

    return result;
}

int FastBoard::count_liberties(const int i, const int k) const {
    if (k==8) {
        return m_libs[i];
    } else if (k==9) {
        return m_libs[m_parent[i]];
    } else if ((k>=4) && (k<=7)){
        int ai = i + m_dirs[k-4];
        return m_libs[m_parent[ai]];
    } else {
        int ai = i + m_dirs[k];
        return m_libs[ai];
    }
}

int FastBoard::get_parent_vertex(const int i) const {
    return m_parent[i];
}

int FastBoard::get_stone(const int i) const {
    return m_stones[i];
}

int FastBoard::get_stonelist_len(const int i) const {
    int len = 0;
    int start = m_parent[i];
    int newpos = start;

    do {
        newpos = m_next[newpos];
        len++;
    } while (newpos != start);

    return len;
}

int FastBoard::get_stonelist_liberties(const int i) const {
    return m_libs[m_parent[i]];
}

int FastBoard::get_lib(const int i) const {
    return m_libs[m_parent[i]];
}

bool StoneList::stonelist_include(const std::vector<StoneList>  & stonelist, const int vertex) {
    for (size_t i = 0; i < stonelist.size(); i++) {
        if (stonelist[i].vertex==vertex) {
            return true;
        }
    }
    return false;
}

int FastBoard::find_1libst(const int i) const {
    int pos = i;

    do {
        if (count_pliberties(pos)==1){
            return pos;
        }
        pos = m_next[pos];
    } while (pos != i);

    return NO_VERTEX;
}

int FastBoard::find_1lib_num(const int i) const {
    int pos = i;
    int count = 0;

    do {
        if (count_pliberties(pos)==1){
            count++;
        }
        pos = m_next[pos];
    } while (pos != i);

    return count;
}


int FastBoard::find_1lib(const int ver_st1lib) const {
    int pos = ver_st1lib;
    do {
        if (count_pliberties(pos)==1){
            for (auto j = 0; j < 4; j++) {
                auto _ai = pos + m_dirs[j];
                if (get_state(_ai)==EMPTY){
                    return _ai;
                }
            }
            return NO_VERTEX;
        }
        pos = m_next[pos];
    } while (pos != ver_st1lib);
    return NO_VERTEX;
}

// only one vertex with 2 libs and these 2 libs is neighbours, will return
/* 2 libs string
type1: 1 vertex with 2 libs and these 2 libs are neighbours
type2: 1 vertex with 1 lib and another 1 vertex with 2 libs
type3: 1 vertex with 2 libs and these 2 libs are face to face
type4: 1 vertex with 1 lib and another 1 vertex with 1 lib

type1 & 2 are ladder shaper
type1 2 libs' pos: plib is 3 and 2 OR plib is 2 and 2
type2 2 libs' pos: plib is 3 and 2
 */

int FastBoard::find_stonelist_twolibs(const int i) const {
    int pos = i;

    do {
        if (count_pliberties(pos)==2){
            auto c0 = get_state(pos + m_dirs[0]);
            auto c1 = get_state(pos + m_dirs[1]);
            auto c2 = get_state(pos + m_dirs[2]);
            auto c3 = get_state(pos + m_dirs[3]);
            if ( (c0==EMPTY && c2==EMPTY) || (c1==EMPTY && c3==EMPTY) ){
                pos = m_next[pos];
                continue;
            }
            return pos;
        }
        pos = m_next[pos];
    } while (pos != i);

    return NO_VERTEX;
}

int FastBoard::find_plibs(const int i, const int libs) const {
    for (auto k = 0; k < 4; k++) {
        auto ai = i + m_dirs[k];
        auto c = get_state(ai);
        if (c==EMPTY && count_pliberties(ai)==libs) {
            return ai;
        }
    }
    return NO_VERTEX;
}

int FastBoard::find_2libst_num(const int vertex, int lib) const {
    int pos = vertex;
    int count = 0;
    do {
        if (count_pliberties(pos)==lib){
            count++;
        }
        pos = m_next[pos];
    } while (pos != vertex);
    return count;
}

int FastBoard::find_2lib_libpos(int vertex, int num) const {
    int pos = vertex;
    int libpos1 = NO_VERTEX;
    int libpos2 = NO_VERTEX;
    do {
        for (auto i = 0; i < 4; i++) {
            auto ai = pos + m_dirs[i];
            if (get_state(ai)==EMPTY) {
                if (libpos1 == NO_VERTEX) {
                    libpos1 = ai;
                } else if (libpos1 != ai) {
                    if (libpos2 == NO_VERTEX) {
                        libpos2 = ai;
                    } else if (libpos2 != ai) {
                        myprintf("2lib string find more than 2 lib\n");
                    } else {
                        continue;
                    }
                } else {
                    continue;
                }
            }
        }
        pos = m_next[pos];
    } while (pos != vertex);

    if (libpos1>libpos2) {
        auto libpos = NO_VERTEX;
        libpos = libpos1;
        libpos1 = libpos2;
        libpos2 = libpos;
    }
    if (num==1) {
        return libpos1;
    } else {
        return libpos2;
    }
}

int FastBoard::get_ladder_escape(int i, int color) const {
    int opp_color = color==BLACK?WHITE:BLACK;
    int escape = NO_VERTEX;
    myprintf("from last_move's neighbours, find his opponent with only 1 liberty\n");
    for (auto k = 0; k < 4; k++) {
        auto ai = i + m_dirs[k];
        auto libs = m_libs[m_parent[ai]];
        auto c = get_state(ai);
        std::string v = move_to_text(ai);
        myprintf("dir_%d %s(%d): %s %d\n", k, v.c_str(),ai, c==WHITE?"W":c==BLACK?"B":c==EMPTY?"EMPTY":"INVAL", libs);
        if ((c==color)&&(libs==1)) {
            myprintf("    only 1 liberty, find escape direction\n");

            int start = ai;
            int newpos = start;
            do {
                for (auto j = 0; j < 4; j++) {
                    auto _ai = newpos + m_dirs[j];
                    auto _libs = m_libs[m_parent[_ai]];
                    auto _c = get_state(_ai);
                    std::string v = move_to_text(_ai);
                    myprintf("    dir_%d %s(%d): %s %d\n", j, v.c_str(),_ai, _c==WHITE?"W":_c==BLACK?"B":_c==EMPTY?"EMPTY":"INVAL", _libs);
                    if (_c==EMPTY) {
                        escape = _ai;
                        myprintf("        escape direction, try to find ladder shape\n");
                        myprintf("        check direction: %s\n", (j==0||j==2)?"1,3":"0,2");
                        for (auto k = 0; k < 4; k++) {
                            auto __ai = _ai + m_dirs[k];
                            auto __libs = m_libs[m_parent[__ai]];
                            auto __c = get_state(__ai);
                            int __carr[4];
                            __carr[k] = __c;
                            std::string v = move_to_text(__ai);
                            myprintf("        dir_%d %s(%d): %s %d\n", k, v.c_str(),__ai, __c==WHITE?"W":__c==BLACK?"B":__c==EMPTY?"EMPTY":"INVAL", __libs);
                            if ((__c==opp_color) && 
                                ( ((j==0||j==2)&&(k==1||k==3)) || ((j==1||j==3)&&(k==0||k==2))) ) {
                                myprintf("        <-- ladder type1 found\n");
                                return escape;
                            }
                            /*
                            if (k==3) {
                                if ((__carr[0]==color && __carr[1]==color && __carr[2]==EMPTY && __carr[3]==EMPTY ) ||
                                    (__carr[1]==color && __carr[2]==color && __carr[3]==EMPTY && __carr[0]==EMPTY ) ||
                                    (__carr[2]==color && __carr[3]==color && __carr[0]==EMPTY && __carr[1]==EMPTY ) ||
                                    (__carr[3]==color && __carr[0]==color && __carr[1]==EMPTY && __carr[2]==EMPTY ) ) {
                                    myprintf("        <-- ladder type2 found\n");
                                    return escape;
                                }
                            }
                            */
                        }
                    }
                }
                newpos = m_next[newpos];
            } while (newpos != start);
        }
    }
    return NO_VERTEX;
}

void FastBoard::print_stonelist(std::vector<StoneList> & stonelist) const {
    myprintf("size: %d\n", stonelist.size());
    myprintf("Note: stone with 1 liberty and 1 liberty who has two EMPTY neighbour\n");
    myprintf("Note: stone with 2 liberty(neighbour) and 2 liberty who has three EMPTY neighbour\n");
    myprintf("\n");
    myprintf("vertex   len  lib  1_stone _2lib  2_stone _3lib\n");
    for (size_t i = 0; i < stonelist.size(); i++) {
        auto vertex = stonelist[i].vertex;
        auto coor = move_to_text(vertex);
        auto ver_1lib = find_1libst(vertex); 
        auto cor_1lib = move_to_text(ver_1lib);
        auto ver_2lib = find_stonelist_twolibs(vertex); 
        auto cor_2lib = move_to_text(ver_2lib);
        if (1 || stonelist[i].lib==1 || stonelist[i].lib==2){
            myprintf("%3s(%3d) %3d %4d %3s(%3d) %5d %3s(%3d) %5d %s\n", 
                coor.c_str(), stonelist[i].vertex, 
                stonelist[i].len, 
                stonelist[i].lib, 
                cor_1lib.c_str(), ver_1lib, 
                find_plibs(ver_1lib, 2),
                cor_2lib.c_str(), ver_2lib, 
                find_plibs(ver_2lib, 3),
                get_string(stonelist[i].vertex).c_str());
        }
    }
}

void FastBoard::print_ladder(std::vector<StoneList> & stonelist) const {
    myprintf("size: %d\n", stonelist.size());
    myprintf("now it will only print 1 lib stonelist, which will be 2 type: 1stone with 1lib or 2stone both has 1lib\n");
    myprintf("\n");
    myprintf("vertex   len    lib ver(1lib_stone) ver(1lib)\n");
    for (size_t i = 0; i < stonelist.size(); i++) {
        auto vertex = stonelist[i].vertex;
        auto coor = move_to_text(vertex);
        auto num_1lib = find_1lib_num(vertex);
        auto ver_1lib = find_1libst(vertex); 
        auto cor_1lib = move_to_text(ver_1lib);

        auto ver_1libpos = NO_VERTEX;
        for (auto j = 0; j < 4; j++) {
            auto _ai = ver_1lib + m_dirs[j];
            if (get_state(_ai)==EMPTY){
                ver_1libpos = _ai;
            }
        }
        auto cor_1libpos = move_to_text(ver_1libpos);

        if (stonelist[i].lib==1) {
            //auto m = is_ladder_escape(ver_1lib, ver_1libpos);
            myprintf("%3s(%3d) %3d %3d(%d)        %3s(%3d)  %3s(%3d) %6s %s\n", 
                coor.c_str(), stonelist[i].vertex, 
                stonelist[i].len, 
                stonelist[i].lib, num_1lib,
                cor_1lib.c_str(), ver_1lib, 
                cor_1libpos.c_str(), ver_1libpos, 
                "ladder",
                get_string(stonelist[i].vertex).c_str());
        }
    }
}

int FastBoard::is_ladder_escape(int ver_st1lib, int ver_1lib) const {
    auto color = get_state(ver_st1lib);
    int opp_color = color==BLACK?WHITE:BLACK;

    // around 1lib stone, find 1lib's direction
    auto i = 0;
    for (i = 0; i < 4; i++) {
        auto ai = ver_st1lib + m_dirs[i];
        if (ai==ver_1lib){
            break;
        }
    }

    // around 1lib, find opp_color, then judge ladder type1 shape
    for (auto j = 0; j < 4; j++) {
        auto ai = ver_1lib + m_dirs[j];
        if (get_state(ai)==opp_color){
            if ( ((j==0||j==2)&&(i==1||i==3)) || ((j==1||j==3)&&(i==0||i==2))) {
                //myprintf("        <-- ladder type1 found\n");
                return ver_1lib;
            }
        }
    }
    return NO_VERTEX;
}

int FastBoard::is_ladder(int vertex) const {
    //auto color = get_state(vertex);
    //auto opp_color = color==WHITE?BLACK:WHITE;

    //now only judge ladder when there is one 1libs stone
    //auto num_1lib = find_1lib_num(vertex);

    auto ver_1lib = find_1libst(vertex); 

    auto ver_1libpos = NO_VERTEX;
    auto dir_1lib = -1;
    for (auto j = 0; j < 4; j++) {
        auto _ai = ver_1lib + m_dirs[j];
        if (get_state(_ai)==EMPTY){
            ver_1libpos = _ai;
            dir_1lib = j;
            break;
        }
    }
    return is_ladder_escape(ver_1lib, ver_1libpos);
}

int FastBoard::find_capture(int escape) const {
    for (auto j = 0; j < 4; j++) {
        auto ai = escape + m_dirs[j];
        if (get_state(ai)==EMPTY && count_pliberties(ai)==3) {
            return ai;
        }
    }
    return NO_VERTEX;
}

int FastBoard::check_ladder_capture(int vertex) const {
    auto color = get_state(vertex);
    auto opp_color = color==WHITE?BLACK:WHITE;

    //check capture's lib after play escape move, capture's lib must >=2
    int count=0;
    int pos = vertex;
    //myprintf("\n");
    do {
        std::string v = move_to_text(pos); 
        //myprintf("%s(%d) opp libs: ", v.c_str(), pos);
        for (auto j = 0; j < 4; j++) {
            auto _ai = pos + m_dirs[j];
            if (get_state(_ai)==opp_color){
                auto opplib = get_lib(_ai);
                if (opplib<2) {
                    auto cor_ai = move_to_text(_ai); 
                    myprintf("%s(%d)-%d\n", cor_ai.c_str(), _ai, opplib);
                    count++;
                }
            }
        }
        pos = m_next[pos];
    } while (pos != vertex);
    //myprintf("\n");
    return count;
}

// escape is already play on the board
int FastBoard::find_escape_libpos(int escape, int num) const {
    //assert num==1 or num==2
    int count=0;
    for (auto i = 0; i < 4; i++) {
        auto ai = escape + m_dirs[i];
        if (get_state(ai)==EMPTY) {
            count++;
            if (count==num) {
                return ai;
            }
        }
    }
    return NO_VERTEX;
}

