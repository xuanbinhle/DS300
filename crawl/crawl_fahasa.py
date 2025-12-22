import requests
import time
import re
from bs4 import BeautifulSoup
from icecream import ic
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

URL_ROOT = "https://www.fahasa.com/sach-trong-nuoc.html"
pages = 40
sess = requests.Session()
sess.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
})

EDGE_DRIVER_PATH = "msedgedriver.exe"
edge_options = Options()
# edge_options.add_argument("--headless") 
edge_options.add_argument(r'--profile-directory=Default')
edge_options.use_chromium = True

def setup(url: str):
    driver = webdriver.Edge(service=Service(executable_path=EDGE_DRIVER_PATH), options=edge_options)
    driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
        "source": """
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            })
        """
    })
    driver.get(url)
    return driver

if __name__ == '__main__':
    driver = setup(URL_ROOT)
    WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "body"))
    )
    html = BeautifulSoup(driver.page_source, "html.parser")
    
    products = html.find("ul", {'id': "products_grid"}).find_all("h2", {'class': "product-name-no-ellipsis p-name-list"})
    ic(len(products))
    for product in products:
        product = product.find("a", href=True)
        title, url = product.text.strip(), product['href']
        ic(title, url)
        
        driver.quit()
        
        time.sleep(1)
        
        url = 'https://www.fahasa.com/khi-hoi-tho-hoa-thinh-khong-tai-ban-2020.html'
        
        driver = setup(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "body"))
        )
        product_html = BeautifulSoup(driver.page_source, "html.parser")
        
        match = re.search(r'PRODUCT_ID\s*=\s*"(\d+)"', driver.page_source)
        if match:
            product_id = match.group(1)

        description = product_html.find("div", {'id': "desc_content", 'class': 'std'}).text.strip()
        ic(description)
        
        detail_table = product_html.find("div", {'id': "product_view_info", 'class': 'content product_view_content'}).find("table")
        details = re.split(r'[\n\t\r]+', detail_table.text.strip())
        details = {details[0::2]: details[1::2]}
        ic(details)
        
        batch_review = []
        reviews = driver.execute_script(f"""
            return fetch('https://www.fahasa.com/fahasa_catalog/product/loadComment', {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
                    'X-Requested-With': 'XMLHttpRequest'
                }},
                body: 'product_id={product_id}&page=1'
            }}).then(res => res.json());
        """)['commtent_list']
        for review in reviews:
            ic(review['name'], review['created_time'], review['content'])

        raise
    