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

#include "config.h"
#include "UCTSearch.h"

#include <boost/format.hpp>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <type_traits>
#include <algorithm>

#include "FastBoard.h"
#include "FastState.h"
#include "FullBoard.h"
#include "GTP.h"
#include "GameState.h"
#include "TimeControl.h"
#include "Timing.h"
#include "Training.h"
#include "Utils.h"
#ifdef USE_OPENCL
#include "OpenCLScheduler.h"
#endif

using namespace Utils;

constexpr int UCTSearch::UNLIMITED_PLAYOUTS;

class OutputAnalysisData {
public:
    OutputAnalysisData(const std::string& move, int visits,
                       float winrate, float policy_prior, std::string pv,
                       float lcb, bool lcb_ratio_exceeded)
    : m_move(move), m_visits(visits), m_winrate(winrate),
      m_policy_prior(policy_prior), m_pv(pv), m_lcb(lcb),
      m_lcb_ratio_exceeded(lcb_ratio_exceeded) {};

    std::string get_info_string(int order) const {
        auto tmp = "info move " + m_move
                 + " visits " + std::to_string(m_visits)
                 + " winrate "
                 + std::to_string(static_cast<int>(m_winrate * 10000))
                 + " prior "
                 + std::to_string(static_cast<int>(m_policy_prior * 10000.0f))
                 + " lcb "
                 + std::to_string(static_cast<int>(std::max(0.0f, m_lcb) * 10000));
        if (order >= 0) {
            tmp += " order " + std::to_string(order);
        }
        tmp += " pv " + m_pv;
        return tmp;
    }

    friend bool operator<(const OutputAnalysisData& a,
                          const OutputAnalysisData& b) {
        if (a.m_lcb_ratio_exceeded && b.m_lcb_ratio_exceeded) {
            if (a.m_lcb != b.m_lcb) {
                return a.m_lcb < b.m_lcb;
            }
        }
        if (a.m_visits == b.m_visits) {
            return a.m_winrate < b.m_winrate;
        }
        return a.m_visits < b.m_visits;
    }

private:
    std::string m_move;
    int m_visits;
    float m_winrate;
    float m_policy_prior;
    std::string m_pv;
    float m_lcb;
    bool m_lcb_ratio_exceeded;
};


UCTSearch::UCTSearch(GameState& g, Network& network, Network& network_aux)
    : m_rootstate(g), m_network(network), m_network_aux(network_aux){
    set_playout_limit(cfg_max_playouts);
    set_visit_limit(cfg_max_visits);

    m_root = std::make_unique<UCTNode>(FastBoard::PASS, 0.0f);
}

bool UCTSearch::advance_to_new_rootstate() {
    if (!m_root || !m_last_rootstate) {
        // No current state
        return false;
    }

    if (m_rootstate.get_komi() != m_last_rootstate->get_komi()) {
        return false;
    }

    auto depth =
        int(m_rootstate.get_movenum() - m_last_rootstate->get_movenum());

    if (depth < 0) {
        return false;
    }


    auto test = std::make_unique<GameState>(m_rootstate);
    for (auto i = 0; i < depth; i++) {
        test->undo_move();
    }

    if (m_last_rootstate->board.get_hash() != test->board.get_hash()) {
        // m_rootstate and m_last_rootstate don't match
        return false;
    }

    // Make sure that the nodes we destroyed the previous move are
    // in fact destroyed.
    while (!m_delete_futures.empty()) {
        m_delete_futures.front().wait_all();
        m_delete_futures.pop_front();
    }

    // Try to replay moves advancing m_root
    for (auto i = 0; i < depth; i++) {
        ThreadGroup tg(thread_pool);

        test->forward_move();
        const auto move = test->get_last_move();

        auto oldroot = std::move(m_root);
        m_root = oldroot->find_child(move);

        // Lazy tree destruction.  Instead of calling the destructor of the
        // old root node on the main thread, send the old root to a separate
        // thread and destroy it from the child thread.  This will save a
        // bit of time when dealing with large trees.
        auto p = oldroot.release();
        tg.add_task([p]() { delete p; });
        m_delete_futures.push_back(std::move(tg));

        if (!m_root) {
            // Tree hasn't been expanded this far
            return false;
        }
        m_last_rootstate->play_move(move);
    }

    assert(m_rootstate.get_movenum() == m_last_rootstate->get_movenum());

    if (m_last_rootstate->board.get_hash() != test->board.get_hash()) {
        // Can happen if user plays multiple moves in a row by same player
        return false;
    }

    return true;
}

void UCTSearch::update_root() {
    // Definition of m_playouts is playouts per search call.
    // So reset this count now.
    m_playouts = 0;

#ifndef NDEBUG
    auto start_nodes = m_root->count_nodes_and_clear_expand_state();
#endif

    if (!advance_to_new_rootstate() || !m_root) {
        m_root = std::make_unique<UCTNode>(FastBoard::PASS, 0.0f);
    }
    // Clear last_rootstate to prevent accidental use.
    m_last_rootstate.reset(nullptr);

    // Check how big our search tree (reused or new) is.
    m_nodes = m_root->count_nodes_and_clear_expand_state();

#ifndef NDEBUG
    if (m_nodes > 0) {
        myprintf("update_root, %d -> %d nodes (%.1f%% reused)\n",
            start_nodes, m_nodes.load(), 100.0 * m_nodes.load() / start_nodes);
    }
#endif
}

float UCTSearch::get_min_psa_ratio() const {
    const auto mem_full = UCTNodePointer::get_tree_size() / static_cast<float>(cfg_max_tree_size);
    // If we are halfway through our memory budget, start trimming
    // moves with very low policy priors.
    if (mem_full > 0.5f) {
        // Memory is almost exhausted, trim more aggressively.
        if (mem_full > 0.95f) {
            // if completely full just stop expansion by returning an impossible number
            if (mem_full >= 1.0f) {
                return 2.0f;
            }
            return 0.01f;
        }
        return 0.001f;
    }
    return 0.0f;
}

SearchResult UCTSearch::play_simulation(GameState & currstate,
                                        UCTNode* const node) {
    const auto color = currstate.get_to_move();
    auto result = SearchResult{};

    node->virtual_loss();

    if (node->expandable()) {
        if (currstate.get_passes() >= 2) {
            auto score = currstate.final_score();
            result = SearchResult::from_score(score);
        } else {
            float eval;
            const auto had_children = node->has_children();
            const auto success =
                node->create_children(m_network, m_network_aux, m_nodes, currstate, eval,
                                      get_min_psa_ratio());
            if (!had_children && success) {
                result = SearchResult::from_eval(eval);
            }
        }
    }

    if (node->has_children() && !result.valid()) {
        UCTNode* next;
        if (&m_network==&m_network_aux){
        next = node->uct_select_child(color, node == m_root.get(), false);
        } else {
        next = node->uct_select_child(color, node == m_root.get());
        }
        auto move = next->get_move();
        auto move_str = currstate.move_to_text(move);
        //myprintf("%s %s\n", color==FastBoard::BLACK?"B":"W", move_str.c_str());

        currstate.play_move(move);
        if (move != FastBoard::PASS && currstate.superko()) {
            next->invalidate();
        } else {
            result = play_simulation(currstate, next);
        }
    }

    if (result.valid()) {
        node->update(result.eval());
    }
    node->virtual_loss_undo();

    return result;
}

void UCTSearch::dump_stats(FastState & state, UCTNode & parent) {
    if (cfg_quiet || !parent.has_children()) {
        return;
    }

    const int color = state.get_to_move();

    auto max_visits = 0;
    for (const auto& node : parent.get_children()) {
        max_visits = std::max(max_visits, node->get_visits());
    }

    // sort children, put best move on top
    parent.sort_children(color, cfg_lcb_min_visit_ratio * max_visits);

    if (parent.get_first_child()->first_visit()) {
        return;
    }

    int movecount = 0;
    for (const auto& node : parent.get_children()) {
        // Always display at least two moves. In the case there is
        // only one move searched the user could get an idea why.
        if (++movecount > 2 && !node->get_visits()) break;

        auto move = state.move_to_text(node->get_move());
        auto tmpstate = FastState{state};
        tmpstate.play_move(node->get_move());
        auto pv = move + " " + get_pv(tmpstate, *node);

        myprintf("%4s -> %7d (V: %5.3f%%) (LCB: %5.2f%%) (N: %5.2f%%) PV: %s\n",
            move.c_str(),
            node->get_visits(),
            node->get_visits() ? node->get_raw_eval(color)*100.0f : 0.0f,
            std::max(0.0f, node->get_eval_lcb(color) * 100.0f),
            node->get_policy() * 100.0f,
            pv.c_str());
    }
    tree_stats(parent);
}

float UCTSearch::dump_stats_kr(FastState & state, UCTNode & parent) {
    if (cfg_quiet || !parent.has_children()) {
        return 0.0f;
    }

    const int color = state.get_to_move();

    auto max_visits = 0;
    for (const auto& node : parent.get_children()) {
        max_visits = std::max(max_visits, node->get_visits());
    }

    // sort children, put best move on top
    parent.sort_children(color, cfg_lcb_min_visit_ratio * max_visits);

    if (parent.get_first_child()->first_visit()) {
        return 0.0f;
    }

    int movecount = 0;
    for (const auto& node : parent.get_children()) {
        // Always display at least two moves. In the case there is
        // only one move searched the user could get an idea why.
        if (++movecount > 2 && !node->get_visits()) break;

        auto move = state.move_to_text(node->get_move());
        auto tmpstate = FastState{ state };
        tmpstate.play_move(node->get_move());
        auto pv = move + " " + get_pv(tmpstate, *node);

        myprintf("%4s -> %7d (V: %5.2f%%) (LCB: %5.2f%%) (N: %5.2f%%) PV: %s\n",
            move.c_str(),
            node->get_visits(),
            node->get_visits() ? node->get_raw_eval(color)*100.0f : 0.0f,
            std::max(0.0f, node->get_eval_lcb(color) * 100.0f),
            node->get_policy() * 100.0f,
            pv.c_str());
        return node->get_raw_eval(color);
    }
    tree_stats(parent);
    return 0.0f;
}

void UCTSearch::output_analysis(FastState & state, UCTNode & parent) {
    // We need to make a copy of the data before sorting
    auto sortable_data = std::vector<OutputAnalysisData>();

    if (!parent.has_children()) {
        return;
    }

    const auto color = state.get_to_move();

    auto max_visits = 0;
    for (const auto& node : parent.get_children()) {
        max_visits = std::max(max_visits, node->get_visits());
    }

    for (const auto& node : parent.get_children()) {
        // Send only variations with visits, unless more moves were
        // requested explicitly.
        if (!node->get_visits()
            && sortable_data.size() >= cfg_analyze_tags.post_move_count()) {
            continue;
        }
        auto move = state.move_to_text(node->get_move());
        auto tmpstate = FastState{state};
        tmpstate.play_move(node->get_move());
        auto rest_of_pv = get_pv(tmpstate, *node);
        auto pv = move + (rest_of_pv.empty() ? "" : " " + rest_of_pv);
        auto move_eval = node->get_visits() ? node->get_raw_eval(color) : 0.0f;
        auto policy = node->get_policy();
        auto lcb = node->get_eval_lcb(color);
        auto visits = node->get_visits();
        // Need at least 2 visits for valid LCB.
        auto lcb_ratio_exceeded = visits > 2 &&
            visits > max_visits * cfg_lcb_min_visit_ratio;
        // Store data in array
        sortable_data.emplace_back(move, visits,
                                   move_eval, policy, pv, lcb, lcb_ratio_exceeded);
    }
    // Sort array to decide order
    std::stable_sort(rbegin(sortable_data), rend(sortable_data));

    auto i = 0;
    // Output analysis data in gtp stream
    for (const auto& node : sortable_data) {
        if (i > 0) {
            gtp_printf_raw(" ");
        }
        gtp_printf_raw(node.get_info_string(i).c_str());
        i++;
    }
    gtp_printf_raw("\n");
}

void UCTSearch::tree_stats(const UCTNode& node) {
    size_t nodes = 0;
    size_t non_leaf_nodes = 0;
    size_t depth_sum = 0;
    size_t max_depth = 0;
    size_t children_count = 0;

    std::function<void(const UCTNode& node, size_t)> traverse =
          [&](const UCTNode& node, size_t depth) {
        nodes += 1;
        non_leaf_nodes += node.get_visits() > 1;
        depth_sum += depth;
        if (depth > max_depth) max_depth = depth;

        for (const auto& child : node.get_children()) {
            if (child.get_visits() > 0) {
                children_count += 1;
                traverse(*(child.get()), depth+1);
            } else {
                nodes += 1;
                depth_sum += depth+1;
                if (depth >= max_depth) max_depth = depth+1;
            }
        }
    };

    traverse(node, 0);

    if (nodes > 0) {
        myprintf("%.1f average depth, %d max depth\n",
                 (1.0f*depth_sum) / nodes, max_depth);
        myprintf("%d non leaf nodes, %.2f average children\n",
                 non_leaf_nodes, (1.0f*children_count) / non_leaf_nodes);
    }
}

bool UCTSearch::should_resign(passflag_t passflag, float besteval) {
    if (passflag & UCTSearch::NORESIGN) {
        // resign not allowed
        return false;
    }

    if (cfg_resignpct == 0) {
        // resign not allowed
        return false;
    }

    const size_t num_intersections = m_rootstate.board.get_boardsize()
                                   * m_rootstate.board.get_boardsize();
    const auto move_threshold = num_intersections / 4;
    const auto movenum = m_rootstate.get_movenum();
    if (movenum <= move_threshold) {
        // too early in game to resign
        //return false;
    }

    const auto color = m_rootstate.board.get_to_move();

    const auto is_default_cfg_resign = cfg_resignpct < 0;
    const auto resign_threshold =
        0.01f * (is_default_cfg_resign ? 10 : cfg_resignpct);
    if (besteval > resign_threshold) {
        // eval > cfg_resign
        return false;
    }

    if ((m_rootstate.get_handicap() > 0)
            && (color == FastBoard::WHITE)
            && is_default_cfg_resign) {
        const auto handicap_resign_threshold =
            resign_threshold / (1 + m_rootstate.get_handicap());

        // Blend the thresholds for the first ~215 moves.
        auto blend_ratio = std::min(1.0f, movenum / (0.6f * num_intersections));
        auto blended_resign_threshold = blend_ratio * resign_threshold
            + (1 - blend_ratio) * handicap_resign_threshold;
        if (besteval > blended_resign_threshold) {
            // Allow lower eval for white in handicap games
            // where opp may fumble.
            return false;
        }
    }

    if (!m_rootstate.is_move_legal(color, FastBoard::RESIGN)) {
        return false;
    }

    return true;
}

int UCTSearch::get_best_move(passflag_t passflag) {
    int color = m_rootstate.board.get_to_move();

    auto max_visits = 0;
    for (const auto& node : m_root->get_children()) {
        max_visits = std::max(max_visits, node->get_visits());
    }

    // Make sure best is first
    m_root->sort_children(color,  cfg_lcb_min_visit_ratio * max_visits);

    // Check whether to randomize the best move proportional
    // to the playout counts, early game only.
    auto movenum = int(m_rootstate.get_movenum());
    if (movenum < cfg_random_cnt) {
        m_root->randomize_first_proportionally();
    }

    auto first_child = m_root->get_first_child();
    assert(first_child != nullptr);

    auto bestmove = first_child->get_move();
    auto besteval = first_child->first_visit() ? 0.5f : first_child->get_raw_eval(color);

    // do we want to fiddle with the best move because of the rule set?
    if (passflag & UCTSearch::NOPASS) {
        // were we going to pass?
        if (bestmove == FastBoard::PASS) {
            UCTNode * nopass = m_root->get_nopass_child(m_rootstate);

            if (nopass != nullptr) {
                myprintf("Preferring not to pass.\n");
                bestmove = nopass->get_move();
                if (nopass->first_visit()) {
                    besteval = 1.0f;
                } else {
                    besteval = nopass->get_raw_eval(color);
                }
            } else {
                myprintf("Pass is the only acceptable move.\n");
            }
        }
    } else if (!cfg_dumbpass) {
        const auto relative_score =
            (color == FastBoard::BLACK ? 1 : -1) * m_rootstate.final_score();
        if (bestmove == FastBoard::PASS) {
            // Either by forcing or coincidence passing is
            // on top...check whether passing loses instantly
            // do full count including dead stones.
            // In a reinforcement learning setup, it is possible for the
            // network to learn that, after passing in the tree, the two last
            // positions are identical, and this means the position is only won
            // if there are no dead stones in our own territory (because we use
            // Trump-Taylor scoring there). So strictly speaking, the next
            // heuristic isn't required for a pure RL network, and we have
            // a commandline option to disable the behavior during learning.
            // On the other hand, with a supervised learning setup, we fully
            // expect that the engine will pass out anything that looks like
            // a finished game even with dead stones on the board (because the
            // training games were using scoring with dead stone removal).
            // So in order to play games with a SL network, we need this
            // heuristic so the engine can "clean up" the board. It will still
            // only clean up the bare necessity to win. For full dead stone
            // removal, kgs-genmove_cleanup and the NOPASS mode must be used.

            // Do we lose by passing?
            if (relative_score < 0.0f) {
                myprintf("Passing loses :-(\n");
                // Find a valid non-pass move.
                UCTNode * nopass = m_root->get_nopass_child(m_rootstate);
                if (nopass != nullptr) {
                    myprintf("Avoiding pass because it loses.\n");
                    bestmove = nopass->get_move();
                    if (nopass->first_visit()) {
                        besteval = 1.0f;
                    } else {
                        besteval = nopass->get_raw_eval(color);
                    }
                } else {
                    myprintf("No alternative to passing.\n");
                }
            } else if (relative_score > 0.0f) {
                myprintf("Passing wins :-)\n");
            } else {
                myprintf("Passing draws :-|\n");
                // Find a valid non-pass move.
                const auto nopass = m_root->get_nopass_child(m_rootstate);
                if (nopass != nullptr && !nopass->first_visit()) {
                    const auto nopass_eval = nopass->get_raw_eval(color);
                    if (nopass_eval > 0.5f) {
                        myprintf("Avoiding pass because there could be a winning alternative.\n");
                        bestmove = nopass->get_move();
                        besteval = nopass_eval;
                    }
                }
                if (bestmove == FastBoard::PASS) {
                    myprintf("No seemingly better alternative to passing.\n");
                }
            }
        } else if (m_rootstate.get_last_move() == FastBoard::PASS) {
            // Opponents last move was passing.
            // We didn't consider passing. Should we have and
            // end the game immediately?

            if (!m_rootstate.is_move_legal(color, FastBoard::PASS)) {
                myprintf("Passing is forbidden, I'll play on.\n");
            // do we lose by passing?
            } else if (relative_score < 0.0f) {
                myprintf("Passing loses, I'll play on.\n");
            } else if (relative_score > 0.0f) {
                myprintf("Passing wins, I'll pass out.\n");
                bestmove = FastBoard::PASS;
            } else {
                myprintf("Passing draws, make it depend on evaluation.\n");
                if (besteval < 0.5f) {
                    bestmove = FastBoard::PASS;
                }
            }
        }
    }

    // if we aren't passing, should we consider resigning?
    if (bestmove != FastBoard::PASS) {
        if (should_resign(passflag, besteval)) {
            myprintf("Eval (%.2f%%) looks bad. Resigning.\n",
                     100.0f * besteval);
            bestmove = FastBoard::RESIGN;
        }
    }

    return bestmove;
}

std::string UCTSearch::get_pv(FastState & state, UCTNode& parent) {
    if (!parent.has_children()) {
        return std::string();
    }

    if (parent.expandable()) {
        // Not fully expanded. This means someone could expand
        // the node while we want to traverse the children.
        // Avoid the race conditions and don't go through the rabbit hole
        // of trying to print things from this node.
        return std::string();
    }

    auto& best_child = parent.get_best_root_child(state.get_to_move());
    if (best_child.first_visit()) {
        return std::string();
    }
    auto best_move = best_child.get_move();
    auto res = state.move_to_text(best_move);

    state.play_move(best_move);

    auto next = get_pv(state, best_child);
    if (!next.empty()) {
        res.append(" ").append(next);
    }
    return res;
}

std::string UCTSearch::get_analysis(int playouts) {
    FastState tempstate = m_rootstate;
    int color = tempstate.board.get_to_move();

    auto pvstring = get_pv(tempstate, *m_root);
    float winrate = 100.0f * m_root->get_raw_eval(color);
    return str(boost::format("Playouts: %d, Win: %5.4f%%, PV: %s")
        % playouts % winrate % pvstring.c_str());
}

bool UCTSearch::is_running() const {
    return m_run && UCTNodePointer::get_tree_size() < cfg_max_tree_size;
}

int UCTSearch::est_playouts_left(int elapsed_centis, int time_for_move) const {
    auto playouts = m_playouts.load();
    const auto playouts_left =
        std::max(0, std::min(m_maxplayouts - playouts,
                             m_maxvisits - m_root->get_visits()));

    // Wait for at least 1 second and 100 playouts
    // so we get a reliable playout_rate.
    if (elapsed_centis < 100 || playouts < 100) {
        return playouts_left;
    }
    const auto playout_rate = 1.0f * playouts / elapsed_centis;
    const auto time_left = std::max(0, time_for_move - elapsed_centis);
    return std::min(playouts_left,
                    static_cast<int>(std::ceil(playout_rate * time_left)));
}

size_t UCTSearch::prune_noncontenders(int color, int elapsed_centis, int time_for_move, bool prune) {
    auto lcb_max = 0.0f;
    auto Nfirst = 0;
    // There are no cases where the root's children vector gets modified
    // during a multithreaded search, so it is safe to walk it here without
    // taking the (root) node lock.
    for (const auto& node : m_root->get_children()) {
        if (node->valid()) {
            const auto visits = node->get_visits();
            if (visits > 0) {
                lcb_max = std::max(lcb_max, node->get_eval_lcb(color));
            }
            Nfirst = std::max(Nfirst, visits);
        }
    }
    const auto min_required_visits =
        Nfirst - est_playouts_left(elapsed_centis, time_for_move);
    auto pruned_nodes = size_t{0};
    for (const auto& node : m_root->get_children()) {
        if (node->valid()) {
            const auto visits = node->get_visits();
            const auto has_enough_visits =
                visits >= min_required_visits;
            // Avoid pruning moves that could have the best lower confidence
            // bound.
            const auto high_winrate = visits > 0 ?
                node->get_raw_eval(color) >= lcb_max : false;
            const auto prune_this_node = !(has_enough_visits || high_winrate);

            if (prune) {
                node->set_active(!prune_this_node);
            }
            if (prune_this_node) {
                ++pruned_nodes;
            }
        }
    }

    assert(pruned_nodes < m_root->get_children().size());
    return pruned_nodes;
}

bool UCTSearch::have_alternate_moves(int elapsed_centis, int time_for_move) {
    if (cfg_timemanage == TimeManagement::OFF) {
        return true;
    }
    auto my_color = m_rootstate.get_to_move();
    // For self play use. Disables pruning of non-contenders to not bias the training data.
    auto prune = cfg_timemanage != TimeManagement::NO_PRUNING;
    auto pruned = prune_noncontenders(my_color, elapsed_centis, time_for_move, prune);
    if (pruned < m_root->get_children().size() - 1) {
        return true;
    }
    // If we cannot save up time anyway, use all of it. This
    // behavior can be overruled by setting "fast" time management,
    // which will cause Leela to quickly respond to obvious/forced moves.
    // That comes at the cost of some playing strength as she now cannot
    // think ahead about her next moves in the remaining time.
    auto tc = m_rootstate.get_timecontrol();
    if (!tc.can_accumulate_time(my_color)
        || m_maxplayouts < UCTSearch::UNLIMITED_PLAYOUTS) {
        if (cfg_timemanage != TimeManagement::FAST) {
            return true;
        }
    }
    // In a timed search we will essentially always exit because
    // the remaining time is too short to let another move win, so
    // avoid spamming this message every move. We'll print it if we
    // save at least half a second.
    if (time_for_move - elapsed_centis > 50) {
        myprintf("%.1fs left, stopping early.\n",
                    (time_for_move - elapsed_centis) / 100.0f);
    }
    return false;
}

bool UCTSearch::stop_thinking(int elapsed_centis, int time_for_move) const {
    return m_playouts >= m_maxplayouts
           || m_root->get_visits() >= m_maxvisits
           || elapsed_centis >= time_for_move;
}

void UCTWorker::operator()() {
    do {
        auto currstate = std::make_unique<GameState>(m_rootstate);
        auto result = m_search->play_simulation(*currstate, m_root);
        if (result.valid()) {
            m_search->increment_playouts();
        }
    } while (m_search->is_running());
}

void UCTSearch::increment_playouts() {
    m_playouts++;
}

int UCTSearch::think_ladder(GameState & game, int color) {
    Time start;
    std::vector<int> move;
    update_root();
    m_rootstate.board.set_to_move(color);
    auto rootnum = m_rootstate.get_movenum();
    myprintf("rootstate movenum %d\n", rootnum);
    m_root->prepare_root_node(m_network, m_network_aux, color, m_nodes, m_rootstate);

    auto game_history = game.get_game_history();
    for (const auto &state : game_history) {
        auto num = state->get_movenum();
        if (num<=rootnum) { continue; }
        auto m = state->get_last_move();
        move.push_back(m);
    }

    std::vector<UCTNode*> n;
    UCTNode* node = m_root.get();
    auto currstate = std::make_unique<GameState>(m_rootstate);
    for (size_t i = 0; i < move.size(); i++) {
        const auto color_i = currstate->get_to_move();
        auto cor_i = currstate->move_to_text(move[i]);
        myprintf("learn %d %s %s\n", i, color_i==FastBoard::BLACK?"B":"W", cor_i.c_str());
        currstate = std::make_unique<GameState>(m_rootstate);
        node = m_root.get();
        n.clear();
        n.push_back(node);
        auto result = SearchResult{};
        for (size_t j = 0; j <= i; j++) {
            //myprintf("%d ", j);
            const auto color = currstate->get_to_move();
            if (node->has_children() && !result.valid()) {
                UCTNode* next;
                next = node->ladder_select_child(color, node, true, move[j]);
                auto move = next->get_move();
                auto cor = currstate->move_to_text(move);
                //myprintf("ladder_select %s %s\n", color==FastBoard::BLACK?"B":"W", cor.c_str());
                currstate->play_move(move);
                n.push_back(next);
                node = next;
            }
        }
        node = n.back();
        if (node->expandable()) {
            myprintf("create_children\n");
            float eval;
            const auto had_children = node->has_children();
            const auto success =
                node->create_children(m_network, m_network_aux, m_nodes, *currstate, eval, get_min_psa_ratio());
            if (!had_children && success) {
                result = SearchResult::from_eval(eval);
            }
            if (result.valid()) {
                //node->update(result.eval());
                node->update(1.0);
                myprintf("update %f v:%d (final)\n", result.eval(), m_root->get_visits());
            }
            n.pop_back();
            while(!n.empty()) {
                node = n.back();
                if (result.valid()) {
                    //node->update(result.eval());
                    node->update(1.1);
                    //myprintf("update %f v:%d\n", result.eval(), m_root->get_visits());
                }
                n.pop_back();
            }
        }
        if (result.valid()) {
            //myprintf("po ++\n");
            increment_playouts();
        }
        //dump_stats(m_rootstate, *m_root);
    }
    myprintf("--------------------\n");

    for (const auto& node : m_root->get_children()) {
        node->set_active(true);
    }

    dump_stats(m_rootstate, *m_root);

    Time elapsed;
    int elapsed_centis = Time::timediff_centis(start, elapsed);
    myprintf("%d visits, %d nodes, %d playouts\n\n",
             m_root->get_visits(),
             m_nodes.load(),
             m_playouts.load(),
             (m_playouts * 100.0) / (elapsed_centis+1));

#ifdef _WIN32
    int pos = cfg_weightsfile.find_last_of('\\');
#else
    int pos = cfg_weightsfile.find_last_of('/');
#endif
    int pos1 = cfg_weightsfile.find(".gz");
    std::string s = "";
    if (pos1 > pos) {
        s = cfg_weightsfile.substr(pos + 1, ((pos1 - pos - 1)>8)?8:(pos1-pos-1));
    }
    else {
        s = cfg_weightsfile.substr(pos + 1, 8);
    }

    auto first_child = m_root->get_first_child();
    float tmp_komi = m_rootstate.get_komi();
    auto act_rate = first_child->get_eval(color);

    myprintf("%s-%s()(%.1f-%.2f%%) %s No. %3d %3.1fs %3s %5d %3.4f%% %3.2f%%\n\n",
        PROGRAM_VERSION, s.c_str(), tmp_komi, act_rate*100.0f,
        (color == FastBoard::BLACK) ? "B" : "W",
        int(m_rootstate.get_movenum()) + 1,
        (elapsed_centis + 1) / 100.0f,
        m_rootstate.move_to_text(first_child->get_move()).c_str(),
        first_child->get_visits(),
        first_child->get_eval(color)*100.0f,
        first_child->get_policy()*100.0f);

    m_last_rootstate = std::make_unique<GameState>(m_rootstate);
    return 0;
}


int UCTSearch::think(int color, passflag_t passflag) {
    // Start counting time for us
    m_rootstate.start_clock(color);

    // set up timing info
    Time start;

    update_root();
    // set side to move
    m_rootstate.board.set_to_move(color);

    auto time_for_move =
        m_rootstate.get_timecontrol().max_time_for_move(
            m_rootstate.board.get_boardsize(),
            color, m_rootstate.get_movenum());

    myprintf("Thinking at most %.1f seconds...\n", time_for_move/100.0f);

    // create a sorted list of legal moves (make sure we
    // play something legal and decent even in time trouble)
    m_root->prepare_root_node(m_network, m_network_aux, color, m_nodes, m_rootstate);

    m_run = true;
    int cpus = cfg_num_threads;
    ThreadGroup tg(thread_pool);
    for (int i = 1; i < cpus; i++) {
        tg.add_task(UCTWorker(m_rootstate, this, m_root.get()));
    }

    auto keeprunning = true;
    auto last_update = 0;
    auto last_output = 0;
    float print_interval = 0;
    do {
        auto currstate = std::make_unique<GameState>(m_rootstate);

        auto result = play_simulation(*currstate, m_root.get());
        if (result.valid()) {
            increment_playouts();
        }

        Time elapsed;
        int elapsed_centis = Time::timediff_centis(start, elapsed);

        if (cfg_analyze_tags.interval_centis() &&
            elapsed_centis - last_output > cfg_analyze_tags.interval_centis()) {
            last_output = elapsed_centis;
            output_analysis(m_rootstate, *m_root);
        }

        // output some stats every few seconds
        // check if we should still search
        if (!cfg_quiet && elapsed_centis - last_update > 250) {
            last_update = elapsed_centis;
            myprintf("%s\n", get_analysis(m_playouts.load()).c_str());
            print_interval++;
            if (print_interval > 20) {
                dump_stats(m_rootstate, *m_root);
                print_interval = 0;
            }
        }
        keeprunning  = is_running();
        keeprunning &= !stop_thinking(elapsed_centis, time_for_move);
        keeprunning &= have_alternate_moves(elapsed_centis, time_for_move);
    } while (keeprunning);

    // Make sure to post at least once.
    if (cfg_analyze_tags.interval_centis() && last_output == 0) {
        output_analysis(m_rootstate, *m_root);
    }

    // Stop the search.
    m_run = false;
    tg.wait_all();

    // Reactivate all pruned root children.
    for (const auto& node : m_root->get_children()) {
        node->set_active(true);
    }

    m_rootstate.stop_clock(color);
    if (!m_root->has_children()) {
        return FastBoard::PASS;
    }

    // Display search info.
    myprintf("\n");
    dump_stats(m_rootstate, *m_root);
    Training::record(m_network, m_rootstate, *m_root);

    Time elapsed;
    int elapsed_centis = Time::timediff_centis(start, elapsed);
    myprintf("%d visits, %d nodes, %d playouts, %.0f n/s\n\n",
             m_root->get_visits(),
             m_nodes.load(),
             m_playouts.load(),
             (m_playouts * 100.0) / (elapsed_centis+1));

#ifdef USE_OPENCL
#ifndef NDEBUG
    myprintf("batch stats: %d %d\n",
        batch_stats.single_evals.load(),
        batch_stats.batch_evals.load()
    );
#endif
#endif
        // zliu
#ifdef _WIN32
        int pos = cfg_weightsfile.find_last_of('\\');
        int pos_aux = cfg_weightsfile_aux.find_last_of('\\');
#else
        int pos = cfg_weightsfile.find_last_of('/');
        int pos_aux = cfg_weightsfile_aux.find_last_of('/');
#endif
        int pos1 = cfg_weightsfile.find(".gz");
        int pos1_aux = cfg_weightsfile_aux.find(".gz");
        std::string s = "";
        std::string s_aux = "";
        if (pos1 > pos) {
            s = cfg_weightsfile.substr(pos + 1, ((pos1 - pos - 1)>8)?8:(pos1-pos-1));
        }
        else {
            s = cfg_weightsfile.substr(pos + 1, 8);
        }
        if (cfg_have_aux) {
            if (pos1_aux > pos_aux) {
                s_aux = cfg_weightsfile_aux.substr(pos_aux + 1, ((pos1_aux-pos_aux-1)>8)?8:(pos1_aux-pos_aux-1));
            }
            else {
                s_aux = cfg_weightsfile_aux.substr(pos_aux + 1, 8);
            }
        }

        auto first_child = m_root->get_first_child();
        float tmp_komi = m_rootstate.get_komi();
        //int color = m_rootstate.board.get_to_move();

    auto movenum = int(m_rootstate.get_movenum());
    auto recov_num = 180; 
    auto new_ra = (cfg_ra*recov_num-8+(1-cfg_ra)*movenum)/(recov_num-8);
    if (cfg_ra==1.0f||new_ra>1.0) new_ra = 1.0f;
    auto tmp_rate = std::atanh(first_child->get_eval(color)*2-1)/new_ra;
    auto act_rate = (1+std::tanh(tmp_rate))/2;

        myprintf("%s-%s(%s)%d(%.1f-%.2f%%) %s No. %3d %3.1fs %3s %5d %3.4f%% %3.2f%%\n\n",
            PROGRAM_VERSION, s.c_str(),s_aux.c_str(),cfg_auxmode,tmp_komi,act_rate*100.0f, 
            (color == FastBoard::BLACK) ? "B" : "W",
            int(m_rootstate.get_movenum()) + 1,
            (elapsed_centis + 1) / 100.0f,
            m_rootstate.move_to_text(first_child->get_move()).c_str(),
            first_child->get_visits(),
            first_child->get_eval(color)*100.0f,
            first_child->get_policy()*100.0f);

    if (color == FastBoard::WHITE) {
        if (cfg_komi!=999.0f && act_rate>=cfg_kmrate && tmp_komi>0) {
            if (tmp_komi<=(cfg_komi/8)) {
                tmp_komi = tmp_komi-cfg_kmstep/8;
            } else if (tmp_komi<=(cfg_komi/4)) {
                tmp_komi = tmp_komi-cfg_kmstep/4;
            } else if (tmp_komi<=(cfg_komi/2)) {
                tmp_komi = tmp_komi-cfg_kmstep/2;
            } else {
                tmp_komi = tmp_komi-cfg_kmstep;
            }
            myprintf("-komi -> %.2f\n", tmp_komi);
            m_rootstate.set_komi(tmp_komi);
        }
    } else {
        if (cfg_komi!=999.0f && act_rate>=cfg_kmrate ) {
            tmp_komi = tmp_komi+cfg_kmstep;
            myprintf("+komi -> %.2f\n", tmp_komi);
            m_rootstate.set_komi(tmp_komi);
        }
    }

    int bestmove = get_best_move(passflag);
    if (cfg_have_aux==false) {
        //myprintf("AuxMode already closed\n");
    } else if ((first_child->get_eval(color)*100.0f)>cfg_aux_recover_rate) {
        myprintf("AuxMode closed, winrate > %.2f%%\n", cfg_aux_recover_rate);
        cfg_have_aux = false;
    }

    // Save the explanation.
    m_think_output =
        str(boost::format("move %d, %c => %s\n%s")
        % m_rootstate.get_movenum()
        % (color == FastBoard::BLACK ? 'B' : 'W')
        % m_rootstate.move_to_text(bestmove).c_str()
        % get_analysis(m_root->get_visits()).c_str());

    // Copy the root state. Use to check for tree re-use in future calls.
    m_last_rootstate = std::make_unique<GameState>(m_rootstate);
    return bestmove;
}

int UCTSearch::think_hp(int color, int max_playout, std::vector<Network::PolicyVertexPair> *nodelist, passflag_t passflag) {
    //myprintf("think_hp: %d\n", color);
    // Start counting time for us
    m_rootstate.start_clock(color);

    // set up timing info
    Time start;

    update_root();
    // set side to move
    m_rootstate.board.set_to_move(color);

    auto time_for_move =
        m_rootstate.get_timecontrol().max_time_for_move(
            m_rootstate.board.get_boardsize(),
            color, m_rootstate.get_movenum());

    //myprintf("HP Thinking at most %.1f seconds...\n", time_for_move/100.0f);

    // create a sorted list of legal moves (make sure we
    // play something legal and decent even in time trouble)
    m_root->prepare_root_node(m_network, m_network_aux, color, m_nodes, m_rootstate);

    m_run = true;
    int cpus = cfg_num_threads;
    cpus = 1;
    ThreadGroup tg(thread_pool);
    for (int i = 1; i < cpus; i++) {
        tg.add_task(UCTWorker(m_rootstate, this, m_root.get()));
    }

    auto keeprunning = true;
    auto last_update = 0;
    auto last_output = 0;
    do {
        auto currstate = std::make_unique<GameState>(m_rootstate);

        auto result = play_simulation(*currstate, m_root.get());
        if (result.valid()) {
            increment_playouts();
        }

        Time elapsed;
        int elapsed_centis = Time::timediff_centis(start, elapsed);

        if (cfg_analyze_tags.interval_centis() &&
            elapsed_centis - last_output > cfg_analyze_tags.interval_centis()) {
            last_output = elapsed_centis;
            //output_analysis(m_rootstate, *m_root);
        }

        // output some stats every few seconds
        // check if we should still search
        if (!cfg_quiet && elapsed_centis - last_update > 250) {
            last_update = elapsed_centis;
            //myprintf("HP %s\n", get_analysis(m_playouts.load()).c_str());
        }
        keeprunning  = is_running();
        keeprunning &= !stop_thinking(elapsed_centis, time_for_move);
        keeprunning &= have_alternate_moves(elapsed_centis, time_for_move);
        keeprunning &= (m_playouts < max_playout);
    } while (keeprunning);

    // Make sure to post at least once.
    if (cfg_analyze_tags.interval_centis() && last_output == 0) {
        //output_analysis(m_rootstate, *m_root);
    }

    // Stop the search.
    m_run = false;
    tg.wait_all();

    // Reactivate all pruned root children.
    for (const auto& node : m_root->get_children()) {
        node->set_active(true);
    }

    m_rootstate.stop_clock(color);
    if (!m_root->has_children()) {
        return FastBoard::PASS;
    }

    // Display search info.
    //myprintf("\n");
    //dump_stats(m_rootstate, *m_root);

    /*
    auto max_visits = 0;
    for (const auto& node : m_root->get_children()) {
        max_visits = std::max(max_visits, node->get_visits());
    }
    myprintf("cfg_lcb_min_visit_ratio: %f\n", cfg_lcb_min_visit_ratio); 
    m_root->sort_children(color, cfg_lcb_min_visit_ratio * max_visits);
    */
    for (const auto& node : m_root->get_children()) {
        nodelist->emplace_back(node->get_policy(), node->get_move());
        /*
        if ((node->get_move()==22) || (node->get_move()==387)){
            myprintf("%d %f\n", node->get_move(), node->get_policy());
        }
        */
        continue;
    }

    Training::record(m_network, m_rootstate, *m_root);

    Time elapsed;
    //int elapsed_centis = Time::timediff_centis(start, elapsed);
    /*
    myprintf("HP %d visits, %d nodes, %d playouts, %.0f n/s\n",
             m_root->get_visits(),
             m_nodes.load(),
             m_playouts.load(),
             (m_playouts * 100.0) / (elapsed_centis+1));
    */

#ifdef USE_OPENCL
#ifndef NDEBUG
    myprintf("batch stats: %d %d\n",
        batch_stats.single_evals.load(),
        batch_stats.batch_evals.load()
    );
#endif
#endif
        // zliu
#ifdef _WIN32
        int pos = cfg_weightsfile.find_last_of('\\');
        int pos_aux = cfg_weightsfile_aux.find_last_of('\\');
#else
        int pos = cfg_weightsfile.find_last_of('/');
        int pos_aux = cfg_weightsfile_aux.find_last_of('/');
#endif
        int pos1 = cfg_weightsfile.find(".gz");
        int pos1_aux = cfg_weightsfile_aux.find(".gz");
        std::string s = "";
        std::string s_aux = "";
        if (pos1 > pos) {
            s = cfg_weightsfile.substr(pos + 1, ((pos1 - pos - 1)>8)?8:(pos1-pos-1));
        }
        else {
            s = cfg_weightsfile.substr(pos + 1, 8);
        }
        if (cfg_have_aux) {
            if (pos1_aux > pos_aux) {
                s_aux = cfg_weightsfile_aux.substr(pos_aux + 1, ((pos1_aux-pos_aux-1)>8)?8:(pos1_aux-pos_aux-1));
            }
            else {
                s_aux = cfg_weightsfile_aux.substr(pos_aux + 1, 8);
            }
        }

        auto first_child = m_root->get_first_child();
        //myprintf("firstchild: %d %f\n", first_child->get_move(), first_child->get_policy());
        //float tmp_komi = m_rootstate.get_komi();
        //int color = m_rootstate.board.get_to_move();

    auto movenum = int(m_rootstate.get_movenum());
    auto recov_num = 180; 
    auto new_ra = (cfg_ra*recov_num-8+(1-cfg_ra)*movenum)/(recov_num-8);
    if (cfg_ra==1.0f||new_ra>1.0) new_ra = 1.0f;
    //auto tmp_rate = std::atanh(first_child->get_eval(color)*2-1)/new_ra;
    //auto act_rate = (1+std::tanh(tmp_rate))/2;
    /*
        myprintf("HP %s-(%s)%d(%.1f-%.2f%%) %s No. %3d %3.1fs %3s %5d %3.4f%% %3.2f%%\n",
            PROGRAM_VERSION, s_aux.c_str(),cfg_auxmode,tmp_komi,act_rate*100.0f, 
            (color == FastBoard::BLACK) ? "B" : "W",
            int(m_rootstate.get_movenum()) + 1,
            (elapsed_centis + 1) / 100.0f,
            m_rootstate.move_to_text(first_child->get_move()).c_str(),
            first_child->get_visits(),
            first_child->get_eval(color)*100.0f,
            first_child->get_policy()*100.0f);
    */
    int bestmove = get_best_move(passflag);
    if (cfg_have_aux==false) {
        //myprintf("AuxMode already closed\n");
    } else if ((first_child->get_eval(color)*100.0f)>cfg_aux_recover_rate) {
        myprintf("AuxMode closed, winrate > %.2f%%\n", cfg_aux_recover_rate);
        cfg_have_aux = false;
    }

    // Save the explanation.
    m_think_output =
        str(boost::format("move %d, %c => %s\n%s")
        % m_rootstate.get_movenum()
        % (color == FastBoard::BLACK ? 'B' : 'W')
        % m_rootstate.move_to_text(bestmove).c_str()
        % get_analysis(m_root->get_visits()).c_str());

    // Copy the root state. Use to check for tree re-use in future calls.
    m_last_rootstate = std::make_unique<GameState>(m_rootstate);
    return bestmove;
}


float UCTSearch::think_kr(int color, passflag_t passflag) {
    // Start counting time for us
    m_rootstate.start_clock(color);

    // set up timing info
    Time start;

    update_root();
    // set side to move
    m_rootstate.board.set_to_move(color);

    auto time_for_move =
        m_rootstate.get_timecontrol().max_time_for_move(
            m_rootstate.board.get_boardsize(),
            color, m_rootstate.get_movenum());

    myprintf("Thinking at most %.1f seconds...\n", time_for_move / 100.0f);

    // create a sorted list of legal moves (make sure we
    // play something legal and decent even in time trouble)
    m_root->prepare_root_node(m_network, m_network_aux, color, m_nodes, m_rootstate);

    m_run = true;
    int cpus = cfg_num_threads;
    ThreadGroup tg(thread_pool);
    for (int i = 1; i < cpus; i++) {
        tg.add_task(UCTWorker(m_rootstate, this, m_root.get()));
    }

    auto keeprunning = true;
    auto last_update = 0;
    auto last_output = 0;
    do {
        auto currstate = std::make_unique<GameState>(m_rootstate);

        auto result = play_simulation(*currstate, m_root.get());
        if (result.valid()) {
            increment_playouts();
        }

        Time elapsed;
        int elapsed_centis = Time::timediff_centis(start, elapsed);

        if (cfg_analyze_tags.interval_centis() &&
            elapsed_centis - last_output > cfg_analyze_tags.interval_centis()) {
            last_output = elapsed_centis;
            output_analysis(m_rootstate, *m_root);
        }

        // output some stats every few seconds
        // check if we should still search
        if (!cfg_quiet && elapsed_centis - last_update > 250) {
            last_update = elapsed_centis;
            myprintf("%s\n", get_analysis(m_playouts.load()).c_str());
        }
        keeprunning = is_running();
        keeprunning &= !stop_thinking(elapsed_centis, time_for_move);
        keeprunning &= have_alternate_moves(elapsed_centis, time_for_move);
    } while (keeprunning);

    // Make sure to post at least once.
    if (cfg_analyze_tags.interval_centis() && last_output == 0) {
        output_analysis(m_rootstate, *m_root);
    }

    // Stop the search.
    m_run = false;
    tg.wait_all();

    // Reactivate all pruned root children.
    for (const auto& node : m_root->get_children()) {
        node->set_active(true);
    }

    m_rootstate.stop_clock(color);
    if (!m_root->has_children()) {
        return FastBoard::PASS;
    }

    // Display search info.
    //myprintf("\n");
    //dump_stats(m_rootstate, *m_root);
    float winrate = dump_stats_kr(m_rootstate, *m_root);
    Training::record(m_network, m_rootstate, *m_root);

    Time elapsed;
    int elapsed_centis = Time::timediff_centis(start, elapsed);
    myprintf("%d visits, %d nodes, %d playouts, %.0f n/s\n\n",
        m_root->get_visits(),
        m_nodes.load(),
        m_playouts.load(),
        (m_playouts * 100.0) / (elapsed_centis + 1));

#ifdef USE_OPENCL
#ifndef NDEBUG
    myprintf("batch stats: %d %d\n",
        batch_stats.single_evals.load(),
        batch_stats.batch_evals.load()
    );
#endif
#endif
    // zliu
#ifdef _WIN32
    int pos = cfg_weightsfile.find_last_of('\\');
#else
    int pos = cfg_weightsfile.find_last_of('/');
#endif
    int pos1 = cfg_weightsfile.find(".gz");
    std::string s = "";
    if (pos1 > pos) {
        s = cfg_weightsfile.substr(pos + 1, pos1 - pos - 1);
    }
    else {
        s = cfg_weightsfile.substr(pos + 1);
    }
    auto first_child = m_root->get_first_child();
    //int color = m_rootstate.board.get_to_move();
    myprintf("%s-%s-%s %s No. %3d %3.1fs %3s %5d %3.4f%% %3.2f%%\n\n",
        "LeelaZero", PROGRAM_VERSION, s.c_str(),
        (color == FastBoard::BLACK) ? "B" : "W",
        int(m_rootstate.get_movenum()) + 1,
        (elapsed_centis + 1) / 100.0f,
        m_rootstate.move_to_text(first_child->get_move()).c_str(),
        first_child->get_visits(),
        first_child->get_eval(color)*100.0f,
        first_child->get_policy()*100.0f);
    int bestmove = get_best_move(passflag);

    // Save the explanation.
    m_think_output =
        str(boost::format("move %d, %c => %s\n%s")
            % m_rootstate.get_movenum()
            % (color == FastBoard::BLACK ? 'B' : 'W')
            % m_rootstate.move_to_text(bestmove).c_str()
            % get_analysis(m_root->get_visits()).c_str());

    // Copy the root state. Use to check for tree re-use in future calls.
    m_last_rootstate = std::make_unique<GameState>(m_rootstate);
    return winrate;
    //return bestmove;
}

// Brief output from last think() call.
std::string UCTSearch::explain_last_think() const {
    return m_think_output;
}

void UCTSearch::ponder() {
    auto disable_reuse = cfg_analyze_tags.has_move_restrictions();
    if (disable_reuse) {
        m_last_rootstate.reset(nullptr);
    }

    update_root();

    m_root->prepare_root_node(m_network, m_network_aux, m_rootstate.board.get_to_move(),
                              m_nodes, m_rootstate);

    m_run = true;
    ThreadGroup tg(thread_pool);
    for (auto i = size_t{1}; i < cfg_num_threads; i++) {
        tg.add_task(UCTWorker(m_rootstate, this, m_root.get()));
    }
    Time start;
    auto keeprunning = true;
    auto last_output = 0;
    do {
        auto currstate = std::make_unique<GameState>(m_rootstate);
        auto result = play_simulation(*currstate, m_root.get());
        if (result.valid()) {
            increment_playouts();
        }
        if (cfg_analyze_tags.interval_centis()) {
            Time elapsed;
            int elapsed_centis = Time::timediff_centis(start, elapsed);
            if (elapsed_centis - last_output > cfg_analyze_tags.interval_centis()) {
                last_output = elapsed_centis;
                output_analysis(m_rootstate, *m_root);
            }
        }
        keeprunning  = is_running();
        keeprunning &= !stop_thinking(0, 1);
    } while (!Utils::input_pending() && keeprunning);

    // Make sure to post at least once.
    if (cfg_analyze_tags.interval_centis() && last_output == 0) {
        output_analysis(m_rootstate, *m_root);
    }

    // Stop the search.
    m_run = false;
    tg.wait_all();

    // Display search info.
    myprintf("\n");
    dump_stats(m_rootstate, *m_root);

    myprintf("\n%d visits, %d nodes\n\n", m_root->get_visits(), m_nodes.load());

    // Copy the root state. Use to check for tree re-use in future calls.
    if (!disable_reuse) {
        m_last_rootstate = std::make_unique<GameState>(m_rootstate);
    }
}

void UCTSearch::set_playout_limit(int playouts) {
    static_assert(std::is_convertible<decltype(playouts),
                                      decltype(m_maxplayouts)>::value,
                  "Inconsistent types for playout amount.");
    m_maxplayouts = std::min(playouts, UNLIMITED_PLAYOUTS);
}

void UCTSearch::set_visit_limit(int visits) {
    static_assert(std::is_convertible<decltype(visits),
                                      decltype(m_maxvisits)>::value,
                  "Inconsistent types for visits amount.");
    // Limit to type max / 2 to prevent overflow when multithreading.
    m_maxvisits = std::min(visits, UNLIMITED_PLAYOUTS);
}

