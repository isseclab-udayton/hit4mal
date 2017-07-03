import requests
import os
from bs4 import BeautifulSoup
from urllib.parse import urljoin

CURRENT_DIR = os.getcwd()
DOWNLOAD_DIR = os.path.join(CURRENT_DIR, 'malware_album')

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

url = 'http://old.vision.ece.ucsb.edu/~lakshman/malware_images/album/'


class Extractor(object):
    """docstring for Parser"""
    def __init__(self, html, base_url):
        self.soup = BeautifulSoup(html, "html5lib")
        self.base_url = base_url

    def get_album(self):
        galaries = self.soup.find("div", {"id": "galleries"})
        table = galaries.find("table")
        families = table.find_all('a', href=True)
        for family in families:
            family_name = family.text.strip()
            if family_name != "":
                yield family_name, urljoin(self.base_url, family['href'])


    def get_image_table(self):
        tables = self.soup.find('table')
        for td in tables.find_all('td'):
            image_atag = td.find('a', href=True)
            if image_atag is not None:
                yield image_atag['href']

    def get_pages(self):
        pages = self.soup.find_all('a', href=True)
        seen = list()
        for page in pages:
            if page is not None:
                if 'index' in page['href']:
                    page_url = page['href']
                    if page_url not in seen:
                        seen.append(page_url)
                        yield page_url

    def get_image_link(self):
        """
        return downloadable image's url
        """
        table = self.soup.find('table')
        image_tag = table.find('img')
        image_name = self.soup.find_all("b")[1].text
        return image_tag['src'], image_name

        # image = td.find_all('img')
        # print(image)
        # if image is not None:
        #     return urljoin(self.base_url, image['src'])



def fetch(image_url, image_name, folder):
    r = requests.get(image_url, stream=True)
    image_file = os.path.join(folder, image_name)
    with open(image_file, 'wb') as f:
        for chunk in r.iter_content(1024):
            f.write(chunk)
    del r


def extract_image(page_html, family_url, folder):
    """
    Extract image from page
    """
    image_extractor = Extractor(page_html, family_url)
    for url in image_extractor.get_image_table():
        image_page_url = urljoin(family_url, url)
        # print(image_page_url)
        imres = requests.get(image_page_url)
        image_page_extractor = Extractor(imres.text, image_page_url)
        image_src, image_name = image_page_extractor.get_image_link()

        image_link = urljoin(image_page_url, image_src)

        print(image_link, image_name)
        # Download image
        fetch(image_link, image_name, folder)



def download(url):
    res = requests.get(url)
    parser = Extractor(res.text, url)
    # for each family, fetch image
    for family, family_url in parser.get_album():
        family_folder = os.path.join(DOWNLOAD_DIR, family)
        print(family_folder)
        os.makedirs(family_folder)
        # print(os.path.join(DOWNLOAD_DIR, family_folder))

        res = requests.get(family_url)
        if res.status_code == 200:
            page_extractor = Extractor(res.text, family_url)
            count = 1
            print('Page ', count)
            extract_image(res.text, family_url, family_folder) # Extract on first page
            for page in page_extractor.get_pages():
                page_url = urljoin(family_url, page)

                count += 1
                print("Page ", count)

                r = requests.get(page_url)
                extract_image(r.text, family_url, family_folder)


            # print('>', image_extractor.get_image_link())
        else:
            print('%s has status code: %s' % (family, res.status_code))





if __name__ == '__main__':
    download(url)



