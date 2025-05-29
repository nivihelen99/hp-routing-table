#pragma once

#include <array>
#include <memory>
#include <vector>
#include <string>
#include <cstdint>
#include <atomic>
#include <shared_mutex>
#include <mutex>  // Added missing mutex header
#include <chrono>
#include <unordered_map>
#include <algorithm>
#include <bit>
#include <cstdio>  // Added for sscanf
#include <cstdlib> // Added for rand()

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
        // Simple parsing - in production, use inet_pton
        uint32_t a, b, c, d;
        if (sscanf(str.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) == 4) {
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
        
        uint32_t mask = ~((1ULL << (32 - prefix_len)) - 1);
        return (addr_ & mask) == (prefix.addr_ & mask);
    }
    
    uint32_t hash() const {
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
    std::array<uint64_t, 2> addr_;  // 128 bits as two 64-bit values

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
        
        for (size_t i = 0; i < prefix_len; ++i) {
            if (get_bit(i) != prefix.get_bit(i)) {
                return false;
            }
        }
        return true;
    }
    
    uint32_t hash() const {
        return std::hash<uint64_t>{}(addr_[0]) ^ std::hash<uint64_t>{}(addr_[1]);
    }
    
    bool operator==(const IPv6Address& other) const {
        return addr_[0] == other.addr_[0] && addr_[1] == other.addr_[1];
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
};

// Statistics Structure - Fixed to allow proper copying
struct RoutingTableStats {
    std::atomic<uint64_t> lookup_count{0};
    std::atomic<uint64_t> insert_count{0};
    std::atomic<uint64_t> delete_count{0};
    std::atomic<uint64_t> total_lookup_time_ns{0};
    std::atomic<size_t> route_count{0};
    std::atomic<size_t> memory_usage_bytes{0};
    
    // Copy constructor - properly handle atomic members
    RoutingTableStats(const RoutingTableStats& other) {
        lookup_count.store(other.lookup_count.load());
        insert_count.store(other.insert_count.load());
        delete_count.store(other.delete_count.load());
        total_lookup_time_ns.store(other.total_lookup_time_ns.load());
        route_count.store(other.route_count.load());
        memory_usage_bytes.store(other.memory_usage_bytes.load());
    }
    
    // Assignment operator - properly handle atomic members
    RoutingTableStats& operator=(const RoutingTableStats& other) {
        if (this != &other) {
            lookup_count.store(other.lookup_count.load());
            insert_count.store(other.insert_count.load());
            delete_count.store(other.delete_count.load());
            total_lookup_time_ns.store(other.total_lookup_time_ns.load());
            route_count.store(other.route_count.load());
            memory_usage_bytes.store(other.memory_usage_bytes.load());
        }
        return *this;
    }
    
    // Default constructor
    RoutingTableStats() = default;
    
    double avg_lookup_time_ns() const {
        uint64_t count = lookup_count.load();
        return count > 0 ? static_cast<double>(total_lookup_time_ns.load()) / count : 0.0;
    }
};

// Memory Pool for efficient allocation
class MemoryPool {
private:
    static constexpr size_t BLOCK_SIZE = 64;  // Cache line size
    static constexpr size_t POOL_SIZE = 1024 * 1024;  // 1MB chunks
    
    struct Block {
        alignas(BLOCK_SIZE) char data[BLOCK_SIZE];
    };
    
    std::vector<std::unique_ptr<Block[]>> pools_;
    std::atomic<size_t> current_pool_index_{0};
    std::atomic<size_t> current_block_index_{0};
    mutable std::mutex allocation_mutex_;

public:
    MemoryPool() {
        add_new_pool();
    }
    
    void* allocate(size_t size) {
        if (size > BLOCK_SIZE) {
            return std::aligned_alloc(BLOCK_SIZE, size);
        }
        
        std::lock_guard<std::mutex> lock(allocation_mutex_);
        
        size_t pool_idx = current_pool_index_.load();
        size_t block_idx = current_block_index_.fetch_add(1);
        
        if (block_idx >= POOL_SIZE / BLOCK_SIZE) {
            add_new_pool();
            pool_idx = current_pool_index_.load();
            block_idx = 0;
            current_block_index_.store(1);
        }
        
        return &pools_[pool_idx][block_idx];
    }
    
    void deallocate(void* ptr, size_t size) {
        if (size > BLOCK_SIZE) {
            std::free(ptr);
        }
        // Pool blocks are not individually freed
    }

private:
    void add_new_pool() {
        pools_.push_back(std::make_unique<Block[]>(POOL_SIZE / BLOCK_SIZE));
        current_pool_index_.store(pools_.size() - 1);
        current_block_index_.store(0);
    }
};

// Patricia Trie Node
template<typename IPAddressType>
class PatriciaNode {
private:
    std::array<std::unique_ptr<PatriciaNode>, 2> children_;
    std::unique_ptr<RouteEntry<IPAddressType>> route_entry_;
    uint32_t bit_position_;
    IPAddressType test_prefix_;
    uint8_t stored_prefix_length_;

public:
    PatriciaNode(uint32_t bit_pos = 0) 
        : bit_position_(bit_pos), stored_prefix_length_(0) {}
    
    ~PatriciaNode() = default;
    
    // Non-copyable, movable
    PatriciaNode(const PatriciaNode&) = delete;
    PatriciaNode& operator=(const PatriciaNode&) = delete;
    PatriciaNode(PatriciaNode&&) = default;
    PatriciaNode& operator=(PatriciaNode&&) = default;
    
    bool insert(const RouteEntry<IPAddressType>& entry) {
        return insert_recursive(entry, 0);
    }
    
    const RouteEntry<IPAddressType>* lookup(const IPAddressType& dest) const {
        const RouteEntry<IPAddressType>* best_match = nullptr;
        lookup_recursive(dest, best_match);
        return best_match;
    }
    
    bool remove(const IPAddressType& prefix, uint8_t prefix_len) {
        return remove_recursive(prefix, prefix_len);
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
        }
        for (const auto& child : children_) {
            if (child) {
                usage += child->memory_usage();
            }
        }
        return usage;
    }

private:
    bool insert_recursive(const RouteEntry<IPAddressType>& entry, uint32_t current_bit) {
        // If this is a leaf or we've reached the prefix length
        if (current_bit >= entry.prefix_length) {
            if (route_entry_) {
                // Route already exists - update it
                *route_entry_ = entry;
                return false;  // Not a new route
            } else {
                route_entry_ = std::make_unique<RouteEntry<IPAddressType>>(entry);
                test_prefix_ = entry.prefix;
                stored_prefix_length_ = entry.prefix_length;
                return true;  // New route added
            }
        }
        
        // Determine which child to follow
        bool bit_value = entry.prefix.get_bit(current_bit);
        size_t child_idx = bit_value ? 1 : 0;
        
        if (!children_[child_idx]) {
            children_[child_idx] = std::make_unique<PatriciaNode>(current_bit + 1);
        }
        
        return children_[child_idx]->insert_recursive(entry, current_bit + 1);
    }
    
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
    
    bool remove_recursive(const IPAddressType& prefix, uint8_t prefix_len) {
        if (route_entry_ && 
            route_entry_->prefix == prefix && 
            route_entry_->prefix_length == prefix_len) {
            route_entry_.reset();
            return true;
        }
        
        // Try children
        for (auto& child : children_) {
            if (child && child->remove_recursive(prefix, prefix_len)) {
                // Clean up empty child nodes
                if (child->is_empty()) {
                    child.reset();
                }
                return true;
            }
        }
        
        return false;
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
    mutable RoutingTableStats stats_;  // Made mutable to allow modification from const methods
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
        auto start_time = std::chrono::high_resolution_clock::now();
        
        bool inserted = false;
        if (config_.thread_safe) {
            std::unique_lock<std::shared_mutex> lock(table_mutex_);
            inserted = root_->insert(entry);
        } else {
            inserted = root_->insert(entry);
        }
        
        if (config_.enable_statistics) {
            stats_.insert_count.fetch_add(1);
            if (inserted) {
                stats_.route_count.fetch_add(1);
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time).count();
            stats_.total_lookup_time_ns.fetch_add(static_cast<uint64_t>(duration));
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
            stats_.delete_count.fetch_add(1);
            stats_.route_count.fetch_sub(1);
        }
        
        return deleted;
    }
    
    const RouteEntry<IPAddressType>* lookup(const IPAddressType& dest) const {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        const RouteEntry<IPAddressType>* result = nullptr;
        if (config_.thread_safe) {
            std::shared_lock<std::shared_mutex> lock(table_mutex_);
            result = root_->lookup(dest);
        } else {
            result = root_->lookup(dest);
        }
        
        if (config_.enable_statistics) {
            stats_.lookup_count.fetch_add(1);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time).count();
            stats_.total_lookup_time_ns.fetch_add(static_cast<uint64_t>(duration));
        }
        
        return result;
    }
    
    size_t bulk_insert(const std::vector<RouteEntry<IPAddressType>>& entries) {
        size_t inserted_count = 0;
        
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
            stats_.insert_count.fetch_add(entries.size());
            stats_.route_count.fetch_add(inserted_count);
        }
        
        return inserted_count;
    }
    
    void optimize_table() {
        if (config_.thread_safe) {
            std::unique_lock<std::shared_mutex> lock(table_mutex_);
        }
        // TODO: Implement route aggregation
        // This would involve traversing the trie and combining adjacent routes
        // with the same next-hop and interface
    }
    
    void compact_memory() {
        // Rebuild the entire trie to eliminate fragmentation
        std::vector<RouteEntry<IPAddressType>> all_routes;
        extract_all_routes(all_routes);
        
        if (config_.thread_safe) {
            std::unique_lock<std::shared_mutex> lock(table_mutex_);
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
        stats = stats_;  // Now works with proper copy constructor/assignment
        if (config_.thread_safe) {
            std::shared_lock<std::shared_mutex> lock(table_mutex_);
            stats.memory_usage_bytes.store(root_->memory_usage());
        } else {
            stats.memory_usage_bytes.store(root_->memory_usage());
        }
    }
    
    void reset_statistics() override {
        stats_.lookup_count.store(0);
        stats_.insert_count.store(0);
        stats_.delete_count.store(0);
        stats_.total_lookup_time_ns.store(0);
    }
    
    // Utility methods
    std::vector<RouteEntry<IPAddressType>> get_routes_for_prefix(
        const IPAddressType& prefix, uint8_t prefix_length) const {
        std::vector<RouteEntry<IPAddressType>> matching_routes;
        
        if (config_.thread_safe) {
            std::shared_lock<std::shared_mutex> lock(table_mutex_);
        }
        
        // TODO: Implement prefix-based route extraction
        return matching_routes;
    }

private:
    void extract_all_routes(std::vector<RouteEntry<IPAddressType>>& routes) const {
        // TODO: Implement recursive route extraction from trie
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
    
    for (size_t i = 0; i < count; ++i) {
        RouteEntry<IPv4Address> route;
        route.prefix = IPv4Address(rand() & 0xFFFFFF00);  // Class C networks
        route.prefix_length = 24;
        route.next_hop = IPv4Address((rand() & 0xFF) << 24 | 0x010101);
        route.interface_id = rand() % 16 + 1;
        route.metric = rand() % 1000;
        route.flags = RouteFlags::DYNAMIC;
        route.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        routes.push_back(route);
    }
    
    return routes;
}

