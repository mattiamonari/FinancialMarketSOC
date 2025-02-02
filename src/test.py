import unittest
import numpy as np
import networkx as nx
from main import G, update_positions, update_price
from market_statistics import calculate_price_returns, calculate_volume_imbalance

class TestMarketSimulation(unittest.TestCase):

    def test_network_structure(self):
        # Test that hedge funds are high-degree nodes
        hedge_funds = [node for node in G.nodes if G.nodes[node]['type'] == 'hedge_fund']
        traders = [node for node in G.nodes if G.nodes[node]['type'] == 'trader']
        degrees_hf = [G.degree[node] for node in hedge_funds]
        degrees_tr = [G.degree[node] for node in traders]
        self.assertGreater(min(degrees_hf), min(degrees_tr), "Hedge funds should have high degrees.")
        self.assertGreater(max(degrees_hf), max(degrees_tr), "Hedge funds should have high degrees.")


    def test_node_attributes(self):
        # Test that all nodes have necessary attributes
        for node in G.nodes:
            self.assertIn('type', G.nodes[node])
            self.assertIn('position', G.nodes[node])
            self.assertIn('trade_size', G.nodes[node])

    def test_update_positions(self):
        # Run position updates and verify changes
        initial_positions = {node: G.nodes[node]['position'] for node in G.nodes}
        update_positions(1)
        updated_positions = {node: G.nodes[node]['position'] for node in G.nodes}
        self.assertNotEqual(initial_positions, updated_positions, "Positions should update after running update_positions.")

    def test_update_price(self):
        # Test price update logic
        prices = [100]
        initial_price = prices[-1]
        prices, _, _ = update_price(prices, [], [])
        new_price = prices[-1]
        self.assertNotEqual(initial_price, new_price, "Price should update after running update_price.")
        self.assertGreaterEqual(new_price, 0, "Price should not be negative.")

    def test_calculate_price_returns(self):
        # Test price return calculation
        sample_prices = [100, 110, 105, 120]
        returns = calculate_price_returns(sample_prices)
        expected_returns = [(110 - 100) / 100, (105 - 110) / 110, (120 - 105) / 105]
        returns = np.array(returns)
        expected_returns = np.array(expected_returns)
        self.assertEqual(returns.all(), expected_returns.all(), "Returns calculation is incorrect.")

    def test_calculate_volume_imbalance(self):
        # Test volume imbalance calculation
        imbalance = calculate_volume_imbalance(G)
        self.assertGreaterEqual(imbalance, 0, "Volume imbalance should be non-negative.")
        self.assertLessEqual(imbalance, 1, "Volume imbalance should not exceed 1.")

if __name__ == '__main__':
    unittest.main()
