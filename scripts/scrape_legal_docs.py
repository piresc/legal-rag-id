import requests
from bs4 import BeautifulSoup
import os
import time

class LegalDocScraper:
    def __init__(self, output_dir="indonesian_laws"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.session = requests.Session()
        self.headers = {
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9,ms;q=0.8,id;q=0.7',
            'Connection': 'keep-alive',
            'Referer': 'https://jdihn.go.id/pencarian?jenis=7',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36',
            'X-Requested-With': 'XMLHttpRequest'
        }
        
    def scrape_direct_url(self, url, filename=None):
        """Scrape a document from direct JDIHN URL"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            
            pdf_link = None
            for link in soup.find_all("a", class_="btn-download"):
                pdf_url = link.get("href")
                if pdf_url and pdf_url.endswith(".pdf"):
                    pdf_link = pdf_url if pdf_url.startswith("http") else f"https://jdihn.go.id{pdf_url}"
                    break
            
            if not pdf_link:
                print("No PDF link found on the page")
                return
                
            filename = filename or os.path.basename(pdf_link)
            self._download_file(pdf_link, filename)
            
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error scraping direct URL: {e}")
        except Exception as e:
            print(f"Error scraping direct URL: {e}")

    def scrape_jdihn(self, query="UU Lalu Lintas", max_docs=10, max_retries=None):
        """Scrape PDFs from JDIHN website with infinite retries"""
        base_url = "https://jdihn.go.id"
        
    def scrape_jdihn_api(self, jenis=7, max_docs=10):
        """Scrape documents using JDIHN API endpoint"""
        api_url = "https://jdihn.go.id/api/search"
        params = {'jenis': jenis}
        
        try:
            response = self.session.get(api_url, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data or not isinstance(data, list):
                print("No valid data returned from API")
                return
                
            for i, doc in enumerate(data[:max_docs]):
                if 'pdf_url' in doc:
                    self._download_file(doc['pdf_url'], f"law_api_{i+1}.pdf")
                    time.sleep(2)
                
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error scraping API: {e}")
        except Exception as e:
            print(f"Error scraping API: {e}")
        
        # Test different endpoint patterns
        endpoint_patterns = [
            f"{base_url}/search?categories=Hukum%20Publik&q={query.replace(' ', '%20')}",
            f"{base_url}/search?q={query.replace(' ', '%20')}",
            f"{base_url}/search?query={query.replace(' ', '%20')}",
            f"{base_url}/api/search?q={query.replace(' ', '%20')}"
        ]
        
        # Find first working endpoint
        search_url = None
        for pattern in endpoint_patterns:
            try:
                head_response = requests.head(pattern, timeout=5)
                if head_response.status_code == 200:
                    search_url = pattern
                    print(f"Using endpoint: {search_url}")
                    break
            except requests.exceptions.RequestException:
                continue
                
        if not search_url:
            print("No working endpoints found")
            return
        
        # Verify URL is accessible before proceeding
        try:
            head_response = requests.head(search_url, timeout=5)
            head_response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"URL verification failed: {search_url}")
            print(f"Error: {str(e)}")
            return
        
        attempt = 0
        while True:
            try:
                response = requests.get(search_url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                
                pdf_links = []
                for link in soup.find_all("a", class_="btn-download"):
                    pdf_url = link.get("href")
                    if pdf_url and pdf_url.endswith(".pdf"):
                        pdf_links.append(pdf_url if pdf_url.startswith("http") else f"{base_url}{pdf_url}")
                
                if not pdf_links:
                    print("No PDF links found on the page")
                    time.sleep(60)  # Wait longer before retrying
                    continue
                
                for i, pdf_url in enumerate(pdf_links[:max_docs]):
                    self._download_file(pdf_url, f"law_{i+1}.pdf")
                    time.sleep(2)  # Be polite with delay
                
                return  # Success
                
            except requests.exceptions.HTTPError as e:
                attempt += 1
                print(f"HTTP Error (attempt {attempt}): {e}")
                if e.response.status_code == 404:
                    print(f"URL not found: {search_url}")
                    return
                time.sleep(min(5 * (attempt + 1), 300))  # Cap delay at 5 minutes
            except Exception as e:
                attempt += 1
                print(f"Error scraping JDIHN (attempt {attempt}): {e}")
                time.sleep(min(5 * (attempt + 1), 300))  # Cap delay at 5 minutes
            
            # Log progress periodically
            if attempt % 10 == 0:
                with open(os.path.join(self.output_dir, "scrape_errors.log"), "a") as log_file:
                    log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Attempt {attempt}: Still trying to scrape {search_url}\n")
    
    def scrape_hukumonline(self, query="", max_docs=10, hierarchy="uu", year=""):
        """Scrape documents from hukumonline.com API"""
        base_url = "https://search.hukumonline.com"
        api_url = f"{base_url}/search/regulations"
        params = {
            'p': 0,
            'l': max_docs,
            'o': 'desc',
            's': 'relevancy',
            'q': query,
            'hierarchy': hierarchy,
            'year': year
        }
        
        try:
            response = self.session.get(api_url, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data or not isinstance(data, dict) or 'hits' not in data:
                print("No valid data returned from API")
                return
                
            for i, doc in enumerate(data['hits']['hits'][:max_docs]):
                if '_source' in doc and 'documentUrl' in doc['_source']:
                    pdf_url = doc['_source']['documentUrl']
                    self._download_file(pdf_url, f"hukumonline_{i+1}.pdf")
                    time.sleep(2)
            
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error scraping hukumonline: {e}")
        except Exception as e:
            print(f"Error scraping hukumonline: {e}")
            
    def _download_file(self, url, filename, max_retries=None):
        """Download and save a file with infinite retries"""
        attempt = 0
        while True:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                with open(os.path.join(self.output_dir, filename), "wb") as f:
                    f.write(response.content)
                print(f"Downloaded: {filename}")
                return  # Success
            except requests.exceptions.HTTPError as e:
                attempt += 1
                print(f"HTTP Error downloading {url} (attempt {attempt}): {e}")
                if e.response.status_code == 404:
                    print(f"File not found: {url}")
                    return
                time.sleep(min(5 * (attempt + 1), 300))  # Cap delay at 5 minutes
            except Exception as e:
                attempt += 1
                print(f"Error downloading {url} (attempt {attempt}): {e}")
                time.sleep(min(5 * (attempt + 1), 300))  # Cap delay at 5 minutes
            
            # Log progress periodically
            if attempt % 10 == 0:
                with open(os.path.join(self.output_dir, "scrape_errors.log"), "a") as log_file:
                    log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Attempt {attempt}: Still trying to download {url}\n")

if __name__ == "__main__":
    scraper = LegalDocScraper()
    # Example usage for direct URL:
    # scraper.scrape_direct_url("https://jdihn.go.id/pencarian/detail/1323939/undang-undang-dasar-(uud)-tahun-1945-dan-amandemen")
    scraper.scrape_jdihn()