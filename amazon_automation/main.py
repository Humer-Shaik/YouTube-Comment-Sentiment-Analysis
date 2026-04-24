"""
Main Orchestrator for Amazon Automation
Coordinates all components to execute the complete automation flow
"""
from browser_manager import BrowserManager
from amazon_searcher import AmazonSearcher
from product_scraper import ProductScraper
from cart_manager import CartManager
from excel_manager import ExcelManager
from test_case_manager import TestCaseManager
from config_reader import ConfigReader
from config import Config
from logger_config import LoggerConfig
import traceback

logger = LoggerConfig.setup_logger(__name__)

class AmazonAutomation:
    """Main automation orchestrator"""
    
    def __init__(self):
        """Initialize automation"""
        self.browser_manager = BrowserManager()
        self.excel_manager = ExcelManager()
        self.test_manager = TestCaseManager()
        self.config_reader = ConfigReader()
        self.logger = logger
        self.driver = None
    
    def run(self):
        """
        Execute complete automation flow
        
        Returns:
            True if successful, False otherwise
        """
        test_id = self.test_manager.start_test("Amazon Product Automation")
        
        try:
            # Initialize browser
            self.logger.info("=" * 60)
            self.logger.info("Starting Amazon Automation")
            self.logger.info("=" * 60)
            
            self.driver = self.browser_manager.initialize_browser()
            self.test_manager.log_step("Browser initialized")
            
            # Read configuration
            config = self.config_reader.read_config()
            self.logger.info(f"Configuration loaded: Product={config['product_name']}, "
                           f"Min Rating={config['min_rating']}, Max Price={config['max_price']}")
            self.test_manager.log_step(f"Configuration loaded: {config}")
            
            # Navigate to Amazon
            self.browser_manager.navigate_to(Config.AMAZON_URL)
            self.test_manager.log_step("Navigated to Amazon website")
            
            # Search for product
            searcher = AmazonSearcher(self.driver)
            if not searcher.search_product(config['product_name']):
                raise Exception("Product search failed")
            self.test_manager.log_step(f"Searched for: {config['product_name']}")
            
            # Apply filters
            searcher.apply_rating_filter(config['min_rating'])
            self.test_manager.log_step(f"Applied rating filter: {config['min_rating']}+")
            
            searcher.apply_price_filter(config['max_price'])
            self.test_manager.log_step(f"Applied price filter: ₹{config['max_price']}")
            
            # Scrape products
            scraper = ProductScraper(self.driver)
            all_products = scraper.get_all_products()
            
            if not all_products:
                raise Exception("No products found after scraping")
            
            self.logger.info(f"Found {len(all_products)} products")
            self.test_manager.log_step(f"Scraped {len(all_products)} products")
            
            # Filter products
            filtered_products = scraper.filter_products_by_rating(all_products, config['min_rating'])
            filtered_products = scraper.filter_products_by_price(filtered_products, config['max_price'])
            
            if not filtered_products:
                raise Exception("No products after filtering")
            
            self.logger.info(f"Filtered to {len(filtered_products)} products")
            self.test_manager.log_step(f"Filtered to {len(filtered_products)} products")
            
            # Select cheapest product
            selected_product = scraper.get_cheapest_product(filtered_products)
            if not selected_product:
                raise Exception("Failed to select cheapest product")
            
            self.test_manager.log_step(f"Selected product: {selected_product['title']} - ₹{selected_product['price']}")
            
            # Click product
            cart = CartManager(self.driver)
            if not cart.click_product(selected_product['url']):
                raise Exception("Failed to click product")
            self.test_manager.log_step("Clicked on selected product")
            
            # Add to cart
            if not cart.add_to_cart():
                raise Exception("Failed to add to cart")
            self.test_manager.log_step("Added product to cart")
            
            # Go to cart
            if not cart.go_to_cart():
                raise Exception("Failed to navigate to cart")
            self.test_manager.log_step("Navigated to shopping cart")
            
            # Proceed to checkout
            if not cart.proceed_to_checkout():
                raise Exception("Failed to proceed to checkout")
            self.test_manager.log_step("Proceeded to checkout")
            
            # Check for email prompt
            if cart.check_for_email_prompt():
                self.test_manager.log_step("Email prompt found")
                
                # Take screenshot
                screenshot_path = cart.take_screenshot_before_email("amazon_automation")
                if screenshot_path:
                    self.test_manager.log_step(f"Screenshot taken: {screenshot_path}")
                else:
                    self.logger.warning("Failed to take screenshot")
            else:
                self.logger.warning("Email prompt not found")
            
            # Generate reports
            self.logger.info("Generating reports...")
            
            # Product report
            product_report_path = self.excel_manager.create_product_report(filtered_products)
            self.test_manager.log_step(f"Product report generated: {product_report_path}")
            
            # Test report
            test_report_path = self.excel_manager.create_test_report(
                self.test_manager.get_test_results()
            )
            self.test_manager.log_step(f"Test report generated: {test_report_path}")
            
            # Print summary
            summary = self.test_manager.get_test_summary()
            self.logger.info("=" * 60)
            self.logger.info("Automation Completed Successfully!")
            self.logger.info(f"Total Products Found: {len(all_products)}")
            self.logger.info(f"Products After Filtering: {len(filtered_products)}")
            self.logger.info(f"Selected Product: {selected_product['title']}")
            self.logger.info(f"Price: ₹{selected_product['price']}")
            self.logger.info(f"Rating: {selected_product['rating']}")
            self.logger.info(f"Product Report: {product_report_path}")
            self.logger.info(f"Test Report: {test_report_path}")
            self.logger.info("=" * 60)
            
            # End test
            self.test_manager.end_test('PASS', 'Automation completed successfully')
            
            return True
            
        except Exception as e:
            self.logger.error(f"Automation failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.test_manager.log_step(f"Error: {str(e)}")
            self.test_manager.end_test('FAIL', f"Automation failed: {str(e)}")
            
            return False
            
        finally:
            # Close browser
            self.browser_manager.close_browser()
            self.logger.info("Browser closed")

def main():
    """Main entry point"""
    try:
        automation = AmazonAutomation()
        success = automation.run()
        
        if success:
            print("\n✓ Automation completed successfully!")
            exit(0)
        else:
            print("\n✗ Automation failed!")
            exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print(f"\n✗ Fatal error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
