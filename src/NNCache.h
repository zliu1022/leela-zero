/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2018 Michael O and contributors

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

#ifndef NNCACHE_H_INCLUDED
#define NNCACHE_H_INCLUDED

#include "config.h"

#include <array>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>

class NNCache {
public:

    // Maximum size of the cache in number of items.
    static constexpr int MAX_CACHE_COUNT = 150'000;

    // Minimum size of the cache in number of items.
    static constexpr int MIN_CACHE_COUNT = 6'000;

    struct Netresult {
        // 19x19 board positions
        std::array<float, NUM_INTERSECTIONS> policy;

        // pass
        float policy_pass;

        // winrate
        float winrate;

        Netresult() : policy_pass(0.0f), winrate(0.0f) {
            policy.fill(0.0f);
        }
    };

    static constexpr size_t ENTRY_SIZE =
          sizeof(Netresult)
        + sizeof(std::uint64_t)
        + sizeof(std::unique_ptr<Netresult>);

    NNCache(int size = MAX_CACHE_COUNT);  // ~ 208MiB

    // Set a reasonable size gives max number of playouts
    void set_size_from_playouts(int max_playouts);

    // Resize NNCache
    void resize(int size);

    // Try and find an existing entry.
<<<<<<< HEAD
    bool lookup(std::uint64_t hash, Network::Netresult & result);

    // Insert a new entry.
    void insert(std::uint64_t hash,
                const Network::Netresult& result);
=======
    bool lookup(std::uint64_t hash, Netresult & result);

    // Insert a new entry.
    void insert(std::uint64_t hash,
                const Netresult& result);
>>>>>>> upstream/master

    // Return the hit rate ratio.
    std::pair<int, int> hit_rate() const {
        return {m_hits, m_lookups};
    }

    void dump_stats();

    // Return the estimated memory consumption of the cache.
    size_t get_estimated_size();
private:
<<<<<<< HEAD
    NNCache(int size = 150000);  // ~ 225MB
=======
>>>>>>> upstream/master

    std::mutex m_mutex;

    size_t m_size;

    // Statistics
    int m_hits{0};
    int m_lookups{0};
    int m_inserts{0};

    struct Entry {
<<<<<<< HEAD
        Entry(const Network::Netresult& r)
            : result(r) {}
        Network::Netresult result;  // ~ 1.5KB
=======
        Entry(const Netresult& r)
            : result(r) {}
        Netresult result;  // ~ 1.4KiB
>>>>>>> upstream/master
    };

    // Map from hash to {features, result}
    std::unordered_map<std::uint64_t, std::unique_ptr<const Entry>> m_cache;
    // Order entries were added to the map.
    std::deque<size_t> m_order;
};

#endif
