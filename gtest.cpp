#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <thread>
#include <future>
#include <random>
#include <chrono>
#include <vector>
#include <set>
#include <algorithm>

#include "hp_route_table.h" 

using namespace std::chrono_literals;
using testing::_;
using testing::Return;
using testing::InSequence;

// Test fixture for IPv4 routing table tests
class IPv4RoutingTableTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.enable_statistics = true;
        config_.thread_safe = true;
        config_.use_memory_pool = true;
        table_ = std::make_unique<HPRoutingTable<IPv4Address>>(config_);
    }

    void TearDown() override {
        table_.reset();
    }

    RouteEntry<IPv4Address> createRoute(const std::string& prefix, uint8_t prefix_len,
                                      const std::string& next_hop, uint32_t interface_id = 1,
                                      uint32_t metric = 100) {
        RouteEntry<IPv4Address> route;
        route.prefix = IPv4Address(prefix);
        route.prefix_length = prefix_len;
        route.next_hop = IPv4Address(next_hop);
        route.interface_id = interface_id;
        route.metric = metric;
        route.flags = RouteFlags::STATIC;
        route.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        return route;
    }

    HPRoutingTable<IPv4Address>::Config config_;
    std::unique_ptr<HPRoutingTable<IPv4Address>> table_;
};

// Test fixture for IPv6 routing table tests
class IPv6RoutingTableTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.enable_statistics = true;
        config_.thread_safe = true;
        table_ = std::make_unique<HPRoutingTable<IPv6Address>>(config_);
    }

    void TearDown() override {
        table_.reset();
    }

    RouteEntry<IPv6Address> createRoute(const std::array<uint64_t, 2>& prefix_addr,
                                      uint8_t prefix_len,
                                      const std::array<uint64_t, 2>& next_hop_addr,
                                      uint32_t interface_id = 1) {
        RouteEntry<IPv6Address> route;
        route.prefix = IPv6Address(prefix_addr);
        route.prefix_length = prefix_len;
        route.next_hop = IPv6Address(next_hop_addr);
        route.interface_id = interface_id;
        route.metric = 100;
        route.flags = RouteFlags::STATIC;
        route.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        return route;
    }

    HPRoutingTable<IPv6Address>::Config config_;
    std::unique_ptr<HPRoutingTable<IPv6Address>> table_;
};

// IPv4Address Tests
class IPv4AddressTest : public ::testing::Test {};

TEST_F(IPv4AddressTest, ConstructorAndBasicOperations) {
    // Default constructor
    IPv4Address addr1;
    EXPECT_EQ(addr1.to_uint32(), 0u);

    // Constructor from uint32_t
    IPv4Address addr2(0xC0A80101); // 192.168.1.1
    EXPECT_EQ(addr2.to_uint32(), 0xC0A80101u);

    // Constructor from string
    IPv4Address addr3("192.168.1.1");
    EXPECT_EQ(addr3.to_uint32(), 0xC0A80101u);
    EXPECT_EQ(addr3.to_string(), "192.168.1.1");

    // Invalid string should result in 0
    IPv4Address addr4("invalid.ip.address");
    EXPECT_EQ(addr4.to_uint32(), 0u);
}

TEST_F(IPv4AddressTest, BitOperations) {
    IPv4Address addr("192.168.1.1"); // 11000000.10101000.00000001.00000001
    
    // Test bit_length
    EXPECT_EQ(IPv4Address::bit_length(), 32u);
    
    // Test get_bit for known positions
    EXPECT_TRUE(addr.get_bit(0));   // MSB of 192 (11000000)
    EXPECT_TRUE(addr.get_bit(1));   // Second bit of 192
    EXPECT_FALSE(addr.get_bit(2));  // Third bit of 192
    EXPECT_FALSE(addr.get_bit(31)); // LSB of last octet (1)
    EXPECT_TRUE(addr.get_bit(31));  // LSB should be 1
    
    // Test out of bounds
    EXPECT_FALSE(addr.get_bit(32));
    EXPECT_FALSE(addr.get_bit(100));
}

TEST_F(IPv4AddressTest, PrefixMatching) {
    IPv4Address addr("192.168.1.100");
    IPv4Address prefix1("192.168.1.0");
    IPv4Address prefix2("192.168.0.0");
    IPv4Address prefix3("10.0.0.0");

    // Test various prefix lengths
    EXPECT_TRUE(addr.matches_prefix(prefix1, 24)); // /24 network match
    EXPECT_TRUE(addr.matches_prefix(prefix2, 16)); // /16 network match
    EXPECT_FALSE(addr.matches_prefix(prefix3, 8)); // Different /8 network
    
    // Test edge cases
    EXPECT_TRUE(addr.matches_prefix(prefix1, 0));  // /0 should always match
    EXPECT_FALSE(addr.matches_prefix(prefix1, 33)); // Invalid prefix length
    EXPECT_TRUE(addr.matches_prefix(addr, 32));    // Exact match
}

TEST_F(IPv4AddressTest, ComparisonOperators) {
    IPv4Address addr1("192.168.1.1");
    IPv4Address addr2("192.168.1.1");
    IPv4Address addr3("192.168.1.2");

    EXPECT_TRUE(addr1 == addr2);
    EXPECT_FALSE(addr1 == addr3);
    EXPECT_TRUE(addr1 < addr3);
    EXPECT_FALSE(addr3 < addr1);
}

TEST_F(IPv4AddressTest, HashFunction) {
    IPv4Address addr1("192.168.1.1");
    IPv4Address addr2("192.168.1.1");
    IPv4Address addr3("192.168.1.2");

    EXPECT_EQ(addr1.hash(), addr2.hash());
    EXPECT_NE(addr1.hash(), addr3.hash());
}

// IPv6Address Tests
class IPv6AddressTest : public ::testing::Test {};

TEST_F(IPv6AddressTest, BasicOperations) {
    std::array<uint64_t, 2> addr_data = {0x2001048860000000ULL, 0x0000000000008888ULL};
    IPv6Address addr(addr_data);
    
    EXPECT_EQ(IPv6Address::bit_length(), 128u);
    EXPECT_TRUE(addr == IPv6Address(addr_data));
}

TEST_F(IPv6AddressTest, BitOperations) {
    std::array<uint64_t, 2> addr_data = {0x8000000000000000ULL, 0x0000000000000001ULL};
    IPv6Address addr(addr_data);
    
    EXPECT_TRUE(addr.get_bit(0));   // MSB should be 1
    EXPECT_FALSE(addr.get_bit(1));  // Second bit should be 0
    EXPECT_TRUE(addr.get_bit(127)); // LSB should be 1
    EXPECT_FALSE(addr.get_bit(128)); // Out of bounds
}

TEST_F(IPv6AddressTest, PrefixMatching) {
    std::array<uint64_t, 2> addr_data = {0x2001048860000000ULL, 0x0000000000008888ULL};
    std::array<uint64_t, 2> prefix_data = {0x2001048800000000ULL, 0x0000000000000000ULL};
    
    IPv6Address addr(addr_data);
    IPv6Address prefix(prefix_data);
    
    EXPECT_TRUE(addr.matches_prefix(prefix, 32)); // Should match on first 32 bits
    EXPECT_TRUE(addr.matches_prefix(prefix, 0));  // /0 should always match
    EXPECT_FALSE(addr.matches_prefix(prefix, 129)); // Invalid prefix length
}

// Routing Table Core Functionality Tests
TEST_F(IPv4RoutingTableTest, BasicInsertAndLookup) {
    // Insert a route
    auto route = createRoute("192.168.1.0", 24, "10.0.0.1");
    EXPECT_TRUE(table_->insert_route(route));
    EXPECT_EQ(table_->size(), 1u);

    // Lookup matching destination
    IPv4Address dest("192.168.1.100");
    const auto* result = table_->lookup(dest);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->next_hop, IPv4Address("10.0.0.1"));
    EXPECT_EQ(result->interface_id, 1u);

    // Lookup non-matching destination
    IPv4Address dest2("10.1.1.1");
    const auto* result2 = table_->lookup(dest2);
    EXPECT_EQ(result2, nullptr);
}

TEST_F(IPv4RoutingTableTest, DuplicateRouteInsertion) {
    auto route1 = createRoute("192.168.1.0", 24, "10.0.0.1", 1, 100);
    auto route2 = createRoute("192.168.1.0", 24, "10.0.0.2", 2, 200);

    EXPECT_TRUE(table_->insert_route(route1));
    EXPECT_EQ(table_->size(), 1u);

    // Insert duplicate route (same prefix/length) - should update existing
    EXPECT_FALSE(table_->insert_route(route2));
    EXPECT_EQ(table_->size(), 1u);

    // Verify the route was updated
    IPv4Address dest("192.168.1.100");
    const auto* result = table_->lookup(dest);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->next_hop, IPv4Address("10.0.0.2"));
    EXPECT_EQ(result->interface_id, 2u);
    EXPECT_EQ(result->metric, 200u);
}

TEST_F(IPv4RoutingTableTest, LongestPrefixMatch) {
    // Insert routes with different prefix lengths
    auto route1 = createRoute("192.168.0.0", 16, "10.0.0.1", 1);  // /16
    auto route2 = createRoute("192.168.1.0", 24, "10.0.0.2", 2);  // /24
    auto route3 = createRoute("192.168.1.128", 25, "10.0.0.3", 3); // /25

    EXPECT_TRUE(table_->insert_route(route1));
    EXPECT_TRUE(table_->insert_route(route2));
    EXPECT_TRUE(table_->insert_route(route3));

    // Test lookup for destination matching /25 (most specific)
    IPv4Address dest1("192.168.1.200");
    const auto* result1 = table_->lookup(dest1);
    ASSERT_NE(result1, nullptr);
    EXPECT_EQ(result1->next_hop, IPv4Address("10.0.0.3"));

    // Test lookup for destination matching /24
    IPv4Address dest2("192.168.1.50");
    const auto* result2 = table_->lookup(dest2);
    ASSERT_NE(result2, nullptr);
    EXPECT_EQ(result2->next_hop, IPv4Address("10.0.0.2"));

    // Test lookup for destination matching only /16
    IPv4Address dest3("192.168.2.1");
    const auto* result3 = table_->lookup(dest3);
    ASSERT_NE(result3, nullptr);
    EXPECT_EQ(result3->next_hop, IPv4Address("10.0.0.1"));
}

TEST_F(IPv4RoutingTableTest, RouteDelection) {
    auto route1 = createRoute("192.168.1.0", 24, "10.0.0.1");
    auto route2 = createRoute("10.0.0.0", 8, "192.168.1.1");

    EXPECT_TRUE(table_->insert_route(route1));
    EXPECT_TRUE(table_->insert_route(route2));
    EXPECT_EQ(table_->size(), 2u);

    // Delete existing route
    EXPECT_TRUE(table_->delete_route(IPv4Address("192.168.1.0"), 24));
    EXPECT_EQ(table_->size(), 1u);

    // Verify route is gone
    IPv4Address dest("192.168.1.100");
    const auto* result = table_->lookup(dest);
    EXPECT_EQ(result, nullptr);

    // Delete non-existing route
    EXPECT_FALSE(table_->delete_route(IPv4Address("172.16.0.0"), 16));
    EXPECT_EQ(table_->size(), 1u);

    // Delete remaining route
    EXPECT_TRUE(table_->delete_route(IPv4Address("10.0.0.0"), 8));
    EXPECT_EQ(table_->size(), 0u);
}

TEST_F(IPv4RoutingTableTest, BulkInsertOperations) {
    std::vector<RouteEntry<IPv4Address>> routes;
    
    // Create multiple routes
    for (int i = 1; i <= 100; ++i) {
        std::string prefix = "192.168." + std::to_string(i) + ".0";
        std::string next_hop = "10.0.0." + std::to_string(i % 255 + 1);
        routes.push_back(createRoute(prefix, 24, next_hop, i));
    }

    size_t inserted = table_->bulk_insert(routes);
    EXPECT_EQ(inserted, 100u);
    EXPECT_EQ(table_->size(), 100u);

    // Verify some routes
    IPv4Address dest1("192.168.50.100");
    const auto* result1 = table_->lookup(dest1);
    ASSERT_NE(result1, nullptr);
    EXPECT_EQ(result1->interface_id, 50u);

    IPv4Address dest2("192.168.1.1");
    const auto* result2 = table_->lookup(dest2);
    ASSERT_NE(result2, nullptr);
    EXPECT_EQ(result2->interface_id, 1u);
}

TEST_F(IPv4RoutingTableTest, DefaultRoute) {
    // Insert default route (0.0.0.0/0)
    auto default_route = createRoute("0.0.0.0", 0, "192.168.1.1");
    auto specific_route = createRoute("192.168.1.0", 24, "10.0.0.1");

    EXPECT_TRUE(table_->insert_route(default_route));
    EXPECT_TRUE(table_->insert_route(specific_route));

    // Test that specific route takes precedence
    IPv4Address dest1("192.168.1.100");
    const auto* result1 = table_->lookup(dest1);
    ASSERT_NE(result1, nullptr);
    EXPECT_EQ(result1->next_hop, IPv4Address("10.0.0.1"));

    // Test that default route is used for non-matching destinations
    IPv4Address dest2("8.8.8.8");
    const auto* result2 = table_->lookup(dest2);
    ASSERT_NE(result2, nullptr);
    EXPECT_EQ(result2->next_hop, IPv4Address("192.168.1.1"));
}

// Statistics Tests
TEST_F(IPv4RoutingTableTest, StatisticsTracking) {
    auto route = createRoute("192.168.1.0", 24, "10.0.0.1");
    table_->insert_route(route);

    RoutingTableStats stats;
    table_->get_statistics(stats);

    EXPECT_EQ(stats.insert_count.load(), 1u);
    EXPECT_EQ(stats.route_count.load(), 1u);
    EXPECT_GT(stats.memory_usage_bytes.load(), 0u);

    // Perform some lookups
    IPv4Address dest("192.168.1.100");
    for (int i = 0; i < 10; ++i) {
        table_->lookup(dest);
    }

    table_->get_statistics(stats);
    EXPECT_EQ(stats.lookup_count.load(), 10u);
    EXPECT_GT(stats.total_lookup_time_ns.load(), 0u);
    EXPECT_GT(stats.avg_lookup_time_ns(), 0.0);

    // Test statistics reset
    table_->reset_statistics();
    table_->get_statistics(stats);
    EXPECT_EQ(stats.lookup_count.load(), 0u);
    EXPECT_EQ(stats.insert_count.load(), 0u);
    EXPECT_EQ(stats.total_lookup_time_ns.load(), 0u);
}

// Thread Safety Tests
TEST_F(IPv4RoutingTableTest, ConcurrentInsertAndLookup) {
    const int num_threads = 4;
    const int operations_per_thread = 1000;
    std::vector<std::future<void>> futures;

    // Insert routes concurrently
    for (int t = 0; t < num_threads; ++t) {
        futures.push_back(std::async(std::launch::async, [this, t, operations_per_thread]() {
            for (int i = 0; i < operations_per_thread; ++i) {
                std::string prefix = std::to_string(t * 50 + (i % 50)) + ".0.0.0";
                std::string next_hop = "10.0." + std::to_string(t) + "." + std::to_string(i % 255);
                auto route = createRoute(prefix, 8, next_hop, t * 1000 + i);
                table_->insert_route(route);
            }
        }));
    }

    // Perform concurrent lookups
    futures.push_back(std::async(std::launch::async, [this, operations_per_thread]() {
        for (int i = 0; i < operations_per_thread * 2; ++i) {
            IPv4Address dest(std::to_string(i % 200) + ".1.1.1");
            table_->lookup(dest); // May or may not find a match
            std::this_thread::sleep_for(1us); // Small delay to create contention
        }
    }));

    // Wait for all threads to complete
    for (auto& future : futures) {
        future.wait();
    }

    // Verify final state
    EXPECT_GT(table_->size(), 0u);
    
    RoutingTableStats stats;
    table_->get_statistics(stats);
    EXPECT_EQ(stats.insert_count.load(), num_threads * operations_per_thread);
    EXPECT_GT(stats.lookup_count.load(), 0u);
}

TEST_F(IPv4RoutingTableTest, ConcurrentInsertAndDelete) {
    std::vector<std::future<void>> futures;
    std::atomic<int> successful_inserts{0};
    std::atomic<int> successful_deletes{0};

    // Concurrent inserts
    futures.push_back(std::async(std::launch::async, [this, &successful_inserts]() {
        for (int i = 0; i < 500; ++i) {
            std::string prefix = "192.168." + std::to_string(i % 100) + ".0";
            auto route = createRoute(prefix, 24, "10.0.0.1", i);
            if (table_->insert_route(route)) {
                successful_inserts++;
            }
            std::this_thread::sleep_for(1us);
        }
    }));

    // Concurrent deletes
    futures.push_back(std::async(std::launch::async, [this, &successful_deletes]() {
        std::this_thread::sleep_for(10ms); // Let some inserts happen first
        for (int i = 0; i < 250; ++i) {
            std::string prefix = "192.168." + std::to_string(i % 100) + ".0";
            if (table_->delete_route(IPv4Address(prefix), 24)) {
                successful_deletes++;
            }
            std::this_thread::sleep_for(2us);
        }
    }));

    for (auto& future : futures) {
        future.wait();
    }

    // Verify consistency
    EXPECT_EQ(table_->size(), successful_inserts.load() - successful_deletes.load());
}

// Performance Tests
class IPv4RoutingTablePerformanceTest : public IPv4RoutingTableTest {
protected:
    void SetUp() override {
        config_.enable_statistics = true;
        config_.thread_safe = false; // Disable for pure performance testing
        table_ = std::make_unique<HPRoutingTable<IPv4Address>>(config_);
    }

    std::vector<RouteEntry<IPv4Address>> generateRandomRoutes(size_t count) {
        std::vector<RouteEntry<IPv4Address>> routes;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint32_t> dist(0, 0xFFFFFFFF);

        for (size_t i = 0; i < count; ++i) {
            RouteEntry<IPv4Address> route;
            route.prefix = IPv4Address(dist(gen) & 0xFFFFFF00); // /24 networks
            route.prefix_length = 24;
            route.next_hop = IPv4Address(dist(gen));
            route.interface_id = i % 1000 + 1;
            route.metric = i % 1000;
            route.flags = RouteFlags::DYNAMIC;
            route.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            routes.push_back(route);
        }
        return routes;
    }

    std::vector<IPv4Address> generateRandomDestinations(size_t count) {
        std::vector<IPv4Address> destinations;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint32_t> dist(0, 0xFFFFFFFF);

        for (size_t i = 0; i < count; ++i) {
            destinations.emplace_back(dist(gen));
        }
        return destinations;
    }
};

TEST_F(IPv4RoutingTablePerformanceTest, LargeScaleInsertion) {
    const size_t num_routes = 100000;
    auto routes = generateRandomRoutes(num_routes);

    auto start = std::chrono::high_resolution_clock::now();
    size_t inserted = table_->bulk_insert(routes);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double routes_per_second = (static_cast<double>(inserted) / duration.count()) * 1000000;

    EXPECT_EQ(inserted, num_routes);
    EXPECT_GT(routes_per_second, 10000); // Should insert at least 10K routes/sec

    std::cout << "Inserted " << inserted << " routes in " << duration.count() 
              << " microseconds (" << routes_per_second << " routes/sec)" << std::endl;
}

TEST_F(IPv4RoutingTablePerformanceTest, LookupPerformance) {
    // Insert routes first
    const size_t num_routes = 50000;
    auto routes = generateRandomRoutes(num_routes);
    table_->bulk_insert(routes);

    // Generate test destinations
    const size_t num_lookups = 100000;
    auto destinations = generateRandomDestinations(num_lookups);

    // Perform lookup benchmark
    auto start = std::chrono::high_resolution_clock::now();
    size_t found_count = 0;
    for (const auto& dest : destinations) {
        if (table_->lookup(dest) != nullptr) {
            found_count++;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double lookups_per_second = (static_cast<double>(num_lookups) / duration.count()) * 1000000;

    EXPECT_GT(lookups_per_second, 100000); // Should perform at least 100K lookups/sec

    std::cout << "Performed " << num_lookups << " lookups in " << duration.count() 
              << " microseconds (" << lookups_per_second << " lookups/sec), found " 
              << found_count << " matches" << std::endl;
}

// Configuration Tests
TEST(HPRoutingTableConfigTest, ThreadSafetyConfiguration) {
    HPRoutingTable<IPv4Address>::Config config;
    config.thread_safe = false;
    HPRoutingTable<IPv4Address> table(config);

    // Basic operations should still work without thread safety
    RouteEntry<IPv4Address> route;
    route.prefix = IPv4Address("192.168.1.0");
    route.prefix_length = 24;
    route.next_hop = IPv4Address("10.0.0.1");

    EXPECT_TRUE(table.insert_route(route));
    EXPECT_NE(table.lookup(IPv4Address("192.168.1.100")), nullptr);
}

TEST(HPRoutingTableConfigTest, StatisticsConfiguration) {
    HPRoutingTable<IPv4Address>::Config config;
    config.enable_statistics = false;
    HPRoutingTable<IPv4Address> table(config);

    RouteEntry<IPv4Address> route;
    route.prefix = IPv4Address("192.168.1.0");
    route.prefix_length = 24;
    route.next_hop = IPv4Address("10.0.0.1");

    table.insert_route(route);
    table.lookup(IPv4Address("192.168.1.100"));

    RoutingTableStats stats;
    table.get_statistics(stats);
    
    // When statistics are disabled, counters should remain at 0
    EXPECT_EQ(stats.lookup_count.load(), 0u);
    EXPECT_EQ(stats.insert_count.load(), 0u);
}

// Memory Pool Tests
TEST(MemoryPoolTest, BasicAllocation) {
    MemoryPool pool;
    
    void* ptr1 = pool.allocate(32);
    void* ptr2 = pool.allocate(64);
    void* ptr3 = pool.allocate(16);
    
    EXPECT_NE(ptr1, nullptr);
    EXPECT_NE(ptr2, nullptr);
    EXPECT_NE(ptr3, nullptr);
    EXPECT_NE(ptr1, ptr2);
    EXPECT_NE(ptr2, ptr3);
    EXPECT_NE(ptr1, ptr3);
    
    // Test large allocation (should use aligned_alloc)
    void* large_ptr = pool.allocate(1024);
    EXPECT_NE(large_ptr, nullptr);
    
    // Cleanup (deallocate should not crash)
    pool.deallocate(ptr1, 32);
    pool.deallocate(ptr2, 64);
    pool.deallocate(ptr3, 16);
    pool.deallocate(large_ptr, 1024);
}

// IPv6 Routing Table Tests
TEST_F(IPv6RoutingTableTest, BasicIPv6Operations) {
    std::array<uint64_t, 2> prefix_addr = {0x2001048800000000ULL, 0x0000000000000000ULL};
    std::array<uint64_t, 2> next_hop_addr = {0xFE80000000000000ULL, 0x0000000000000001ULL};
    
    auto route = createRoute(prefix_addr, 32, next_hop_addr);
    EXPECT_TRUE(table_->insert_route(route));
    EXPECT_EQ(table_->size(), 1u);

    // Test lookup
    std::array<uint64_t, 2> dest_addr = {0x2001048860000000ULL, 0x0000000000008888ULL};
    IPv6Address dest(dest_addr);
    
    const auto* result = table_->lookup(dest);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->next_hop, IPv6Address(next_hop_addr));
}

// Edge Cases and Error Handling
TEST_F(IPv4RoutingTableTest, EdgeCases) {
    // Test with maximum prefix length
    auto route32 = createRoute("192.168.1.1", 32, "10.0.0.1");
    EXPECT_TRUE(table_->insert_route(route32));
    
    // Exact match should work
    const auto* result = table_->lookup(IPv4Address("192.168.1.1"));
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->next_hop, IPv4Address("10.0.0.1"));
    
    // Slightly different address should not match
    const auto* result2 = table_->lookup(IPv4Address("192.168.1.2"));
    EXPECT_EQ(result2, nullptr);

    // Test with minimum prefix length (default route)
    auto route0 = createRoute("0.0.0.0", 0, "192.168.1.1");
    EXPECT_TRUE(table_->insert_route(route0));
    
    // Any address should match default route if no better match
    const auto* result3 = table_->lookup(IPv4Address("8.8.8.8"));
    ASSERT_NE(result3, nullptr);
    EXPECT_EQ(result3->next_hop, IPv4Address("192.168.1.1"));
}

