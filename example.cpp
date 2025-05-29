// Example usage and testing
#include <iostream>
#include <random>
#include <chrono>

#include "hp_route_table.h"

void example_usage() {
    std::cout << "High-Performance Routing Table Example\n";
    std::cout << "=====================================\n";
    
    // Create IPv4 routing table
    HPRoutingTable<IPv4Address> table;
    
    // Add some test routes
    RouteEntry<IPv4Address> route1;
    route1.prefix = IPv4Address("192.168.1.0");
    route1.prefix_length = 24;
    route1.next_hop = IPv4Address("10.0.0.1");
    route1.interface_id = 1;
    route1.metric = 100;
    
    table.insert_route(route1);
    
    RouteEntry<IPv4Address> route2;
    route2.prefix = IPv4Address("10.0.0.0");
    route2.prefix_length = 8;
    route2.next_hop = IPv4Address("192.168.1.1");
    route2.interface_id = 2;
    route2.metric = 200;
    
    table.insert_route(route2);
    
    // Test lookups
    auto dest1 = IPv4Address("192.168.1.10");
    const auto* result1 = table.lookup(dest1);
    if (result1) {
        std::cout << "Lookup " << dest1.to_string() << " -> " 
                  << result1->next_hop.to_string() << "\n";
    }
    
    auto dest2 = IPv4Address("10.1.1.1");
    const auto* result2 = table.lookup(dest2);
    if (result2) {
        std::cout << "Lookup " << dest2.to_string() << " -> " 
                  << result2->next_hop.to_string() << "\n";
    }
    
    // Performance test
    std::cout << "\nPerformance Test:\n";
    auto routes = HPRoutingTableUtils::generate_random_routes<IPv4Address>(10000);
    table.bulk_insert(routes);
    
    RoutingTableStats stats;
    table.get_statistics(stats);
    
    std::cout << "Routes in table: " << table.size() << "\n";
    std::cout << "Average lookup time: " << stats.avg_lookup_time_ns() << " ns\n";
    std::cout << "Memory usage: " << stats.memory_usage_bytes.load() << " bytes\n";
}


int main()
{
    example_usage();
}

