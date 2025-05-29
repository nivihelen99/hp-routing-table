#pragma once

#include <array>
#include <memory>
#include <vector>
#include <string>
#include <cstdint>
#include <atomic>
#include <shared_mutex>
#include <mutex>
#include <chrono>
#include <unordered_map>
#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <iomanip>

// Forward declarations
template<typename T> class HPRoutingTable;
template<typename T> class PatriciaNode;

// IP Address Types
class IPv4Address {
private:
    uint32_t addr_;

public:
    IPv4Address() : addr_(0) {}
    explicit IPv4Address(uint32_t addr) : addr_(addr) {}
    explicit IPv4Address(const std::string& str) {
        uint32_t a, b, c, d;
        if (sscanf(str.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) == 4 &&
            a <= 255 && b <= 255 && c <= 255 && d <= 255) {
            addr_ = (a << 24) | (b << 16) | (c << 8) | d;
        } else {
            addr_ = 0;
        }
    }

    uint32_t to_uint32() const { return addr_; }
    static constexpr size_t bit_length() { return 32; }
    
    bool get_bit(size_t position) const {
        if (position >= 32) return false;
        return (addr_ >> (31 - position)) & 1;
    }
    
    bool matches_prefix(const IPv4Address& prefix, uint8_t prefix_len) const {
        if (prefix_len == 0) return true;
        if (prefix_len > 32) return false;
        
        // Fixed potential overflow issue
        if (prefix_len == 32) {
            return addr_ == prefix.addr_;
        }
        
        uint32_t mask = ~((1ULL << (32 - prefix_len)) - 1);
        return (addr_ & mask) == (prefix.addr_ & mask);
    }
    
    size_t hash() const {
        return std::hash<uint32_t>{}(addr_);
    }
    
    std::string to_string() const {
        uint32_t a = (addr_ >> 24) & 0xFF;
        uint32_t b = (addr_ >> 16) & 0xFF;
        uint32_t c = (addr_ >> 8) & 0xFF;
        uint32_t d = addr_ & 0xFF;
        return std::to_string(a) + "." + std::to_string(b) + "." + 
               std::to_string(c) + "." + std::to_string(d);
    }
    
    bool operator==(const IPv4Address& other) const {
        return addr_ == other.addr_;
    }
    
    bool operator<(const IPv4Address& other) const {
        return addr_ < other.addr_;
    }
};

class IPv6Address {
private:
    std::array<uint64_t, 2> addr_;

public:
    IPv6Address() : addr_{0, 0} {}
    explicit IPv6Address(const std::array<uint64_t, 2>& addr) : addr_(addr) {}
    
    static constexpr size_t bit_length() { return 128; }
    
    bool get_bit(size_t position) const {
        if (position >= 128) return false;
        size_t idx = position / 64;
        size_t bit_pos = 63 - (position % 64);
        return (addr_[idx] >> bit_pos) & 1;
    }
    
    bool matches_prefix(const IPv6Address& prefix, uint8_t prefix_len) const {
        if (prefix_len == 0) return true;
        if (prefix_len > 128) return false;
        
        // Optimized bit-by-bit comparison for large prefix lengths
        size_t full_bytes = prefix_len / 8;
        size_t remaining_bits = prefix_len % 8;
        
        // Compare full 64-bit chunks
        for (size_t i = 0; i < 2 && full_bytes > 0; ++i) {
            size_t bytes_in_chunk = std::min(full_bytes, size_t(8));
            uint64_t mask = (bytes_in_chunk == 8) ? ~0ULL : 
                           (~0ULL << (8 * (8 - bytes_in_chunk)));
            
            if ((addr_[i] & mask) != (prefix.addr_[i] & mask)) {
                return false;
            }
            full_bytes -= bytes_in_chunk;
        }
        
        // Handle remaining bits
        if (remaining_bits > 0) {
            size_t bit_position = prefix_len - remaining_bits;
            for (size_t i = 0; i < remaining_bits; ++i) {
                if (get_bit(bit_position + i) != prefix.get_bit(bit_position + i)) {
                    return false;
                }
            }
        }
        
        return true;
    }
    
    size_t hash() const {
        return std::hash<uint64_t>{}(addr_[0]) ^ 
               (std::hash<uint64_t>{}(addr_[1]) << 1);
    }
    
    bool operator==(const IPv6Address& other) const {
        return addr_[0] == other.addr_[0] && addr_[1] == other.addr_[1];
    }
    
    bool operator<(const IPv6Address& other) const {
        return addr_[0] < other.addr_[0] || 
               (addr_[0] == other.addr_[0] && addr_[1] < other.addr_[1]);
    }
};

// Route Entry Structure
enum class RouteFlags : uint32_t {
    STATIC = 1 << 0,
    DYNAMIC = 1 << 1,
    ECMP = 1 << 2,
    AGGREGATE = 1 << 3
};

template<typename IPAddressType>
struct RouteEntry {
    IPAddressType prefix;
    uint8_t prefix_length;
    IPAddressType next_hop;
    uint32_t interface_id;
    uint32_t metric;
    RouteFlags flags;
    uint64_t timestamp;
    
    // ECMP support
    std::vector<IPAddressType> ecmp_next_hops;
    std::vector<uint32_t> ecmp_weights;
    
    RouteEntry() : prefix_length(0), interface_id(0), metric(0), 
                   flags(RouteFlags::STATIC), timestamp(0) {}
    
    // Add copy constructor and assignment operator for proper copying
    RouteEntry(const RouteEntry&) = default;
    RouteEntry& operator=(const RouteEntry&) = default;
    RouteEntry(RouteEntry&&) = default;
    RouteEntry& operator=(RouteEntry&&) = default;
};

// Statistics Structure - Fixed atomics handling
struct RoutingTableStats {
    std::atomic<uint64_t> lookup_count{0};
    std::atomic<uint64_t> insert_count{0};
    std::atomic<uint64_t> delete_count{0};
    std::atomic<uint64_t> total_lookup_time_ns{0};
    std::atomic<size_t> route_count{0};
    std::atomic<size_t> memory_usage_bytes{0};
    
    // Fixed copy constructor
    RoutingTableStats(const RoutingTableStats& other) {
        lookup_count.store(other.lookup_count.load(std::memory_order_relaxed), 
                          std::memory_order_relaxed);
        insert_count.store(other.insert_count.load(std::memory_order_relaxed), 
                          std::memory_order_relaxed);
        delete_count.store(other.delete_count.load(std::memory_order_relaxed), 
                          std::memory_order_relaxed);
        total_lookup_time_ns.store(other.total_lookup_time_ns.load(std::memory_order_relaxed), 
                                  std::memory_order_relaxed);
        route_count.store(other.route_count.load(std::memory_order_relaxed), 
                         std::memory_order_relaxed);
        memory_usage_bytes.store(other.memory_usage_bytes.load(std::memory_order_relaxed), 
                                std::memory_order_relaxed);
    }
    
    // Fixed assignment operator
    RoutingTableStats& operator=(const RoutingTableStats& other) {
        if (this != &other) {
            lookup_count.store(other.lookup_count.load(std::memory_order_relaxed), 
                              std::memory_order_relaxed);
            insert_count.store(other.insert_count.load(std::memory_order_relaxed), 
                              std::memory_order_relaxed);
            delete_count.store(other.delete_count.load(std::memory_order_relaxed), 
                              std::memory_order_relaxed);
            total_lookup_time_ns.store(other.total_lookup_time_ns.load(std::memory_order_relaxed), 
                                      std::memory_order_relaxed);
            route_count.store(other.route_count.load(std::memory_order_relaxed), 
                             std::memory_order_relaxed);
            memory_usage_bytes.store(other.memory_usage_bytes.load(std::memory_order_relaxed), 
                                    std::memory_order_relaxed);
        }
        return *this;
    }
    
    RoutingTableStats() = default;
    
    double avg_lookup_time_ns() const {
        uint64_t count = lookup_count.load(std::memory_order_relaxed);
        return count > 0 ? static_cast<double>(total_lookup_time_ns.load(std::memory_order_relaxed)) / count : 0.0;
    }
};

// Improved Memory Pool
class MemoryPool {
private:
    static constexpr size_t BLOCK_SIZE = 64;
    static constexpr size_t BLOCKS_PER_POOL = 16384; // 1MB pools
    
    struct alignas(BLOCK_SIZE) Block {
        char data[BLOCK_SIZE];
    };
    
    struct Pool {
        std::unique_ptr<Block[]> blocks;
        std::atomic<size_t> next_block{0};
        
        Pool() : blocks(std::make_unique<Block[]>(BLOCKS_PER_POOL)) {}
    };
    
    std::vector<std::unique_ptr<Pool>> pools_;
    std::atomic<size_t> current_pool_{0};
    mutable std::mutex pool_mutex_;

public:
    MemoryPool() {
        pools_.push_back(std::make_unique<Pool>());
    }
    
    void* allocate(size_t size) {
        if (size > BLOCK_SIZE) {
            return std::aligned_alloc(BLOCK_SIZE, ((size + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
        }
        
        size_t pool_idx = current_pool_.load(std::memory_order_acquire);
        auto& pool = pools_[pool_idx];
        
        size_t block_idx = pool->next_block.fetch_add(1, std::memory_order_acq_rel);
        
        if (block_idx >= BLOCKS_PER_POOL) {
            std::lock_guard<std::mutex> lock(pool_mutex_);
            // Double-check after acquiring lock
            if (current_pool_.load() == pool_idx) {
                pools_.push_back(std::make_unique<Pool>());
                current_pool_.store(pools_.size() - 1, std::memory_order_release);
            }
            return allocate(size); // Retry with new pool
        }
        
        return &pool->blocks[block_idx];
    }
    
    void deallocate(void* ptr, size_t size) {
        if (size > BLOCK_SIZE) {
            std::free(ptr);
        }
        // Pool blocks are not individually freed
    }
    
    size_t get_memory_usage() const {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        return pools_.size() * BLOCKS_PER_POOL * BLOCK_SIZE;
    }
};

// Improved Patricia Trie Node
template<typename IPAddressType>
class PatriciaNode {
private:
    std::array<std::unique_ptr<PatriciaNode>, 2> children_;
    std::unique_ptr<RouteEntry<IPAddressType>> route_entry_;
    IPAddressType test_prefix_;
    uint8_t stored_prefix_length_;
    uint32_t bit_position_;
    
public:
    PatriciaNode(uint32_t bit_pos = 0) 
        : stored_prefix_length_(0), bit_position_(bit_pos) {}
    
    ~PatriciaNode() = default;
    
    // Non-copyable, movable
    PatriciaNode(const PatriciaNode&) = delete;
    PatriciaNode& operator=(const PatriciaNode&) = delete;
    PatriciaNode(PatriciaNode&&) = default;
    PatriciaNode& operator=(PatriciaNode&&) = default;
    
    bool insert(const RouteEntry<IPAddressType>& entry) {
        // Check if this node should store the route
        if (bit_position_ >= entry.prefix_length) {
            if (route_entry_) {
                // Update existing route
                *route_entry_ = entry;
                return false; // Not a new route
            } else {
                // Insert new route
                route_entry_ = std::make_unique<RouteEntry<IPAddressType>>(entry);
                test_prefix_ = entry.prefix;
                stored_prefix_length_ = entry.prefix_length;
                return true;
            }
        }
        
        // Determine which child to follow
        bool bit_value = entry.prefix.get_bit(bit_position_);
        size_t child_idx = bit_value ? 1 : 0;
        
        if (!children_[child_idx]) {
            children_[child_idx] = std::make_unique<PatriciaNode>(bit_position_ + 1);
        }
        
        return children_[child_idx]->insert(entry);
    }
    
    const RouteEntry<IPAddressType>* lookup(const IPAddressType& dest) const {
        const RouteEntry<IPAddressType>* best_match = nullptr;
        lookup_recursive(dest, best_match);
        return best_match;
    }
    
    bool remove(const IPAddressType& prefix, uint8_t prefix_len) {
        if (route_entry_ && 
            route_entry_->prefix == prefix && 
            route_entry_->prefix_length == prefix_len) {
            route_entry_.reset();
            return true;
        }
        
        // Try children
        for (auto& child : children_) {
            if (child && child->remove(prefix, prefix_len)) {
                // Clean up empty child nodes
                if (child->is_empty()) {
                    child.reset();
                }
                return true;
            }
        }
        
        return false;
    }
    
    void collect_all_routes(std::vector<RouteEntry<IPAddressType>>& routes) const {
        if (route_entry_) {
            routes.push_back(*route_entry_);
        }
        for (const auto& child : children_) {
            if (child) {
                child->collect_all_routes(routes);
            }
        }
    }
    
    void collect_routes_for_prefix(const IPAddressType& prefix, uint8_t prefix_len,
                                  std::vector<RouteEntry<IPAddressType>>& routes) const {
        if (route_entry_ && 
            route_entry_->prefix.matches_prefix(prefix, prefix_len)) {
            routes.push_back(*route_entry_);
        }
        
        for (const auto& child : children_) {
            if (child) {
                child->collect_routes_for_prefix(prefix, prefix_len, routes);
            }
        }
    }
    
    size_t count_routes() const {
        size_t count = route_entry_ ? 1 : 0;
        for (const auto& child : children_) {
            if (child) {
                count += child->count_routes();
            }
        }
        return count;
    }
    
    size_t memory_usage() const {
        size_t usage = sizeof(PatriciaNode);
        if (route_entry_) {
            usage += sizeof(RouteEntry<IPAddressType>);
            usage += route_entry_->ecmp_next_hops.size() * sizeof(IPAddressType);
            usage += route_entry_->ecmp_weights.size() * sizeof(uint32_t);
        }
        for (const auto& child : children_) {
            if (child) {
                usage += child->memory_usage();
            }
        }
        return usage;
    }

private:
    void lookup_recursive(const IPAddressType& dest, 
                         const RouteEntry<IPAddressType>*& best_match) const {
        // Check if current node has a matching route
        if (route_entry_ && 
            dest.matches_prefix(test_prefix_, stored_prefix_length_)) {
            best_match = route_entry_.get();
        }
        
        // Continue search in appropriate child
        if (bit_position_ < dest.bit_length()) {
            bool bit_value = dest.get_bit(bit_position_);
            size_t child_idx = bit_value ? 1 : 0;
            
            if (children_[child_idx]) {
                children_[child_idx]->lookup_recursive(dest, best_match);
            }
        }
    }
    
    bool is_empty() const {
        return !route_entry_ && !children_[0] && !children_[1];
    }
};

// Base Interface
class IRoutingTable {
public:
    virtual ~IRoutingTable() = default;
    virtual size_t size() const = 0;
    virtual void get_statistics(RoutingTableStats& stats) const = 0;
    virtual void reset_statistics() = 0;
};

// Main High-Performance Routing Table Implementation
template<typename IPAddressType>
class HPRoutingTable : public IRoutingTable {
public:
    struct Config {
        bool enable_aggregation = false;
        bool enable_ecmp = false;
        size_t initial_capacity = 10000;
        bool enable_statistics = true;
        bool thread_safe = true;
        bool use_memory_pool = true;
    };

private:
    std::unique_ptr<PatriciaNode<IPAddressType>> root_;
    mutable std::shared_mutex table_mutex_;
    std::unique_ptr<MemoryPool> memory_pool_;
    mutable RoutingTableStats stats_;
    Config config_;

public:
    explicit HPRoutingTable(const Config& config = {}) 
        : root_(std::make_unique<PatriciaNode<IPAddressType>>()), 
          config_(config) {
        if (config_.use_memory_pool) {
            memory_pool_ = std::make_unique<MemoryPool>();
        }
    }
    
    bool insert_route(const RouteEntry<IPAddressType>& entry) {
        auto start_time = config_.enable_statistics ? 
            std::chrono::high_resolution_clock::now() : 
            std::chrono::high_resolution_clock::time_point{};
        
        bool inserted = false;
        if (config_.thread_safe) {
            std::unique_lock<std::shared_mutex> lock(table_mutex_);
            inserted = root_->insert(entry);
        } else {
            inserted = root_->insert(entry);
        }
        
        if (config_.enable_statistics) {
            stats_.insert_count.fetch_add(1, std::memory_order_relaxed);
            if (inserted) {
                stats_.route_count.fetch_add(1, std::memory_order_relaxed);
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time).count();
            stats_.total_lookup_time_ns.fetch_add(static_cast<uint64_t>(duration), 
                                                 std::memory_order_relaxed);
        }
        
        return inserted;
    }
    
    bool delete_route(const IPAddressType& prefix, uint8_t prefix_length) {
        bool deleted = false;
        if (config_.thread_safe) {
            std::unique_lock<std::shared_mutex> lock(table_mutex_);
            deleted = root_->remove(prefix, prefix_length);
        } else {
            deleted = root_->remove(prefix, prefix_length);
        }
        
        if (config_.enable_statistics && deleted) {
            stats_.delete_count.fetch_add(1, std::memory_order_relaxed);
            stats_.route_count.fetch_sub(1, std::memory_order_relaxed);
        }
        
        return deleted;
    }
    
    const RouteEntry<IPAddressType>* lookup(const IPAddressType& dest) const {
        auto start_time = config_.enable_statistics ? 
            std::chrono::high_resolution_clock::now() : 
            std::chrono::high_resolution_clock::time_point{};
        
        const RouteEntry<IPAddressType>* result = nullptr;
        if (config_.thread_safe) {
            std::shared_lock<std::shared_mutex> lock(table_mutex_);
            result = root_->lookup(dest);
        } else {
            result = root_->lookup(dest);
        }
        
        if (config_.enable_statistics) {
            stats_.lookup_count.fetch_add(1, std::memory_order_relaxed);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time).count();
            stats_.total_lookup_time_ns.fetch_add(static_cast<uint64_t>(duration), 
                                                 std::memory_order_relaxed);
        }
        
        return result;
    }
    
    size_t bulk_insert(const std::vector<RouteEntry<IPAddressType>>& entries) {
        size_t inserted_count = 0;
        
        auto start_time = config_.enable_statistics ? 
            std::chrono::high_resolution_clock::now() : 
            std::chrono::high_resolution_clock::time_point{};
        
        if (config_.thread_safe) {
            std::unique_lock<std::shared_mutex> lock(table_mutex_);
            for (const auto& entry : entries) {
                if (root_->insert(entry)) {
                    inserted_count++;
                }
            }
        } else {
            for (const auto& entry : entries) {
                if (root_->insert(entry)) {
                    inserted_count++;
                }
            }
        }
        
        if (config_.enable_statistics) {
            stats_.insert_count.fetch_add(entries.size(), std::memory_order_relaxed);
            stats_.route_count.fetch_add(inserted_count, std::memory_order_relaxed);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time).count();
            stats_.total_lookup_time_ns.fetch_add(static_cast<uint64_t>(duration), 
                                                 std::memory_order_relaxed);
        }
        
        return inserted_count;
    }
    
    void compact_memory() {
        std::vector<RouteEntry<IPAddressType>> all_routes;
        
        if (config_.thread_safe) {
            std::shared_lock<std::shared_mutex> read_lock(table_mutex_);
            root_->collect_all_routes(all_routes);
            read_lock.unlock();
            
            std::unique_lock<std::shared_mutex> write_lock(table_mutex_);
            root_ = std::make_unique<PatriciaNode<IPAddressType>>();
            for (const auto& route : all_routes) {
                root_->insert(route);
            }
        } else {
            root_->collect_all_routes(all_routes);
            root_ = std::make_unique<PatriciaNode<IPAddressType>>();
            for (const auto& route : all_routes) {
                root_->insert(route);
            }
        }
    }
    
    // IRoutingTable implementation
    size_t size() const override {
        if (config_.thread_safe) {
            std::shared_lock<std::shared_mutex> lock(table_mutex_);
            return root_->count_routes();
        } else {
            return root_->count_routes();
        }
    }
    
    void get_statistics(RoutingTableStats& stats) const override {
        stats = stats_;
        if (config_.thread_safe) {
            std::shared_lock<std::shared_mutex> lock(table_mutex_);
            stats.memory_usage_bytes.store(root_->memory_usage(), std::memory_order_relaxed);
        } else {
            stats.memory_usage_bytes.store(root_->memory_usage(), std::memory_order_relaxed);
        }
    }
    
    void reset_statistics() override {
        stats_.lookup_count.store(0, std::memory_order_relaxed);
        stats_.insert_count.store(0, std::memory_order_relaxed);
        stats_.delete_count.store(0, std::memory_order_relaxed);
        stats_.total_lookup_time_ns.store(0, std::memory_order_relaxed);
    }
    
    // Utility methods
    std::vector<RouteEntry<IPAddressType>> get_routes_for_prefix(
        const IPAddressType& prefix, uint8_t prefix_length) const {
        std::vector<RouteEntry<IPAddressType>> matching_routes;
        
        if (config_.thread_safe) {
            std::shared_lock<std::shared_mutex> lock(table_mutex_);
            root_->collect_routes_for_prefix(prefix, prefix_length, matching_routes);
        } else {
            root_->collect_routes_for_prefix(prefix, prefix_length, matching_routes);
        }
        
        return matching_routes;
    }
    
    std::vector<RouteEntry<IPAddressType>> get_all_routes() const {
        std::vector<RouteEntry<IPAddressType>> all_routes;
        
        if (config_.thread_safe) {
            std::shared_lock<std::shared_mutex> lock(table_mutex_);
            root_->collect_all_routes(all_routes);
        } else {
            root_->collect_all_routes(all_routes);
        }
        
        return all_routes;
    }
};

// Type aliases for convenience
using IPv4RoutingTable = HPRoutingTable<IPv4Address>;
using IPv6RoutingTable = HPRoutingTable<IPv6Address>;

// Utility functions for benchmarking and testing
namespace HPRoutingTableUtils {
    
    template<typename IPAddressType>
    std::vector<RouteEntry<IPAddressType>> generate_random_routes(size_t count);
    
    template<typename IPAddressType>
    std::vector<IPAddressType> generate_random_destinations(size_t count);
    
    template<typename IPAddressType>
    void benchmark_lookup_performance(const HPRoutingTable<IPAddressType>& table,
                                    const std::vector<IPAddressType>& destinations,
                                    size_t iterations = 1000000);
    
    // Load routes from file (BGP dump format)
    std::vector<RouteEntry<IPv4Address>> load_bgp_v4_routes(const std::string& filename);
    std::vector<RouteEntry<IPv6Address>> load_bgp_v6_routes(const std::string& filename);
}

// Template specializations for IPv4
template<>
inline std::vector<RouteEntry<IPv4Address>> 
HPRoutingTableUtils::generate_random_routes<IPv4Address>(size_t count) {
    std::vector<RouteEntry<IPv4Address>> routes;
    routes.reserve(count);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> addr_dist(0, 0xFFFFFFFF);
    std::uniform_int_distribution<uint8_t> prefix_dist(8, 32);
    std::uniform_int_distribution<uint32_t> interface_dist(1, 16);
    std::uniform_int_distribution<uint32_t> metric_dist(0, 1000);
    
    for (size_t i = 0; i < count; ++i) {
        RouteEntry<IPv4Address> route;
        route.prefix = IPv4Address(addr_dist(gen) & 0xFFFFFF00);
        route.prefix_length = prefix_dist(gen);
        route.next_hop = IPv4Address(addr_dist(gen));
        route.interface_id = interface_dist(gen);
        route.metric = metric_dist(gen);
        route.flags = RouteFlags::DYNAMIC;
        route.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        routes.push_back(route);
    }
    
    return routes;
}

template<>
inline std::vector<IPv4Address> 
HPRoutingTableUtils::generate_random_destinations<IPv4Address>(size_t count) {
    std::vector<IPv4Address> destinations;
    destinations.reserve(count);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> addr_dist(0, 0xFFFFFFFF);
    
    for (size_t i = 0; i < count; ++i) {
        destinations.emplace_back(addr_dist(gen));
    }
    
    return destinations;
}

template<> 
inline void HPRoutingTableUtils::benchmark_lookup_performance<IPv4Address>(
    const HPRoutingTable<IPv4Address>& table, 
    const std::vector<IPv4Address>& destinations, 
    size_t iterations) 
{
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < iterations; ++i) {
        const auto& dest = destinations[i % destinations.size()];
        volatile auto result = table.lookup(dest);
        // Using volatile to prevent compiler optimization of the lookup call
        (void)result; // Suppress unused variable warning
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Calculate and display performance metrics
    double avg_lookup_time = static_cast<double>(duration.count()) / iterations;
    double lookups_per_second = 1000000.0 / avg_lookup_time; // Convert from microseconds
    
    std::cout << "Routing Table Lookup Performance Results:\n";
    std::cout << "Total iterations: " << iterations << "\n";
    std::cout << "Total time: " << duration.count() << " microseconds\n";
    std::cout << "Average lookup time: " << std::fixed << std::setprecision(3) 
              << avg_lookup_time << " microseconds\n";
    std::cout << "Lookups per second: " << std::fixed << std::setprecision(0) 
              << lookups_per_second << "\n";
}

