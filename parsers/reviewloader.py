# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 08:44:31 2020.

@author: Odegov Ilya
"""

import os
import sys
from logging import getLogger, INFO
from json import dumps, loads
# from random import shuffle
import re

from bs4 import BeautifulSoup
from pandas import DataFrame
from tqdm import tqdm


from ekatalogparser import get_product_reviews

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils import get_page, log

PARSER_LOGGER = getLogger("logger.e-katalog.reviewloader")
NUMBER_PATTERN = re.compile(r"[0-9]{1,}")


def _write_to_csv(products_reviews):
    """Write recived reviews to csv file."""
    products_reviews = DataFrame(products_reviews).sample(frac=1)
    products_reviews.to_csv("reviews.csv", index=None, encoding="utf-8")


def _join_with_pattern_url(pattern, additional_parts):
    """Create new list of valid urls."""
    new_urls = [
        "".join((pattern, part))
        for part in additional_parts
    ]
    return new_urls


def _get_main_menu_items_url(html):
    """Get item url for next next transition."""
    main_menu = html.find(name="div", class_="mainmenu ff-roboto")
    main_menu_items = main_menu.find_all(name="li", class_="mainmenu-item")
    main_menu_items_url = [
        item.find(name="a", class_="mainmenu-link  ").attrs["href"]
        for item in main_menu_items
    ]
    main_menu_items_url[0] = main_menu_items[0].find(
        name="a", class_="mainmenu-link mainmenu-link-first ").attrs["href"]
    return main_menu_items_url


def _get_item_categories_url(item_url):
    """Get all categories from separate main menu item."""
    page = get_page(item_url, PARSER_LOGGER)
    if not page:
        return []
    html = BeautifulSoup(page.content, 'xml')
    if not html:
        return []
    categories = html.find_all(name="div", class_="subcat-title2")
    categories_url = []
    for category in categories:
        urls = category.find_all("A")
        if not urls:
            urls = category.find_all("a")
        if not urls:
            continue
        urls = [
            url.attrs["href"]
            for url in urls if url.attrs.get("href", None)
        ]
        categories_url.extend(urls)
    return categories_url


def _get_all_categories_url(main_menu_items_url):
    """Loop by main menu items.

    Get all categories from item and save their to saparate list.

    """
    all_categories_url = []
    try:
        loader = tqdm(
            main_menu_items_url, desc="Load category from menu items:",
            total=len(main_menu_items_url)
        )
        for item_url in loader:
            item_categories_url = _get_item_categories_url(item_url)
            all_categories_url.extend(item_categories_url)
    # if push ctrl+c save already recieve category url
    except KeyboardInterrupt:
        return all_categories_url
    return all_categories_url


def _preprocess_category_url(url):
    """Process input url.

    Saparate category number from input url and create url in format
    list/specific_category_number.
    """
    if "list" in url:
        return url
    category_number = re.findall(NUMBER_PATTERN, url)
    if category_number:
        category_number = category_number[0]
        return "/" + "/".join(("list", category_number))
    return None


def _preprocess_categories_url(categories_url):
    """Preprocess lof by all categories."""
    preprocess_categories_url = []
    for category_url in categories_url:
        preprocess_category_url = _preprocess_category_url(category_url)
        if preprocess_category_url:
            preprocess_categories_url.append(preprocess_category_url)
    return preprocess_categories_url


def _get_url_to_products_review_urls(html):
    """Get all url of products reviews."""
    products_review_urls = []
    products = html.find_all(
        name="div", class_="ib model-short-links no-mobile")
    for product in products:
        product = product.find(name="A", class_=None)
        if product:
            products_review_urls.append(product.attrs["link"])
    return products_review_urls


def _get_product_review_urls_from_category(category_url):
    """Get url to page with all products from recieve catogory."""
    page = get_page(category_url, PARSER_LOGGER)
    if not page:
        return []
    html = BeautifulSoup(page.content, 'xml')
    if not html:
        return []
    # get quantity of pages with products from category
    try:
        product_page_num = int(
            html.find("div", class_="ib page-num").find_all("a")[-1].text
            )
    except AttributeError:
        product_page_num = 0
    product_page_num = (
        int(product_page_num * 0.3) if product_page_num <= 200 else 20
    )
    products_review_urls = _get_url_to_products_review_urls(html)
    # scrape 30 percent of products from category
    for i in range(1, product_page_num + 1):
        next_url = category_url + f"/{i}/"
        page = get_page(next_url, None)
        if not page:
            continue
        html = BeautifulSoup(page.content, 'xml')
        if not html:
            continue
        next_products_review_urls = _get_url_to_products_review_urls(html)
        products_review_urls.extend(next_products_review_urls)
    return products_review_urls


def _get_products_review_urls_from_categories(categories_url):
    """
    Loop by categories of the product.

    Get url for a product page and save it to the list.

    """
    products_review_urls = []
    try:
        loader = tqdm(
            categories_url, total=len(categories_url),
            desc="Ger products review's urls from specific category:"
        )
        for category_url in loader:
            product_review_urls = _get_product_review_urls_from_category(
                category_url)
            products_review_urls.extend(product_review_urls)
    # if push ctrl+c save already recieve products review urls
    except KeyboardInterrupt:
        return products_review_urls
    return products_review_urls


def _get_urls_to_product_review():
    """Get as much as possible products reviews url."""
    main_url = "https://www.e-katalog.ru"
    page = get_page(main_url, PARSER_LOGGER)
    html = BeautifulSoup(page.content, 'xml')
    # get list of valid items urls from main page
    main_menu_items_url = _get_main_menu_items_url(html)
    main_menu_items_url = _join_with_pattern_url(main_url, main_menu_items_url)
    with open("main_items.json", "w") as fout:
        data = dumps(main_menu_items_url)
        fout.write(data)
    # get list of valid urls for all categories from site
    categories_url = _get_all_categories_url(main_menu_items_url)
    categories_url = _preprocess_categories_url(categories_url)
    categories_url = _join_with_pattern_url(main_url, categories_url)
    with open("categories.json", "w") as fout:
        data = dumps(categories_url)
        fout.write(data)
    # get urls for each products page from every category.
    products_review_urls = _get_products_review_urls_from_categories(
        categories_url
    )
    products_review_urls = _join_with_pattern_url(
        main_url, products_review_urls
    )
    with open("products_reviews.json", "w") as fout:
        data = dumps(products_review_urls)
        fout.write(data)
    return main_menu_items_url, categories_url, products_review_urls


def get_products_reviwes():
    """Get as mutch as possibbe product reviews."""
    try:
        with open("products_reviews.json", "r") as fin:
            product_review_urls = loads(fin.read())
    except FileExistsError:
        product_review_urls = _get_urls_to_product_review()
    # titles, comments, pluses, minuses, helpfuls+, helpfuls-, scores
    products_reviews = {
        "title": [], "comment": [], "plus": [], "minus": [],
        "helpful+": [], "helpful-": [], "score": []
    }
    try:
        loader = tqdm(
            product_review_urls, total=len(product_review_urls),
            desc="Load review from product url."
        )
        for product_review_url in loader:
            try:
                product_reviews = get_product_reviews(product_review_url)
                for i, key in enumerate(products_reviews.keys()):
                    products_reviews[key].extend(product_reviews[i])
            except Exception as err:
                print(err)
                continue
    except KeyboardInterrupt:
        _write_to_csv(products_reviews)

    _write_to_csv(products_reviews)


if __name__ == "__main__":
    _ = _get_urls_to_product_review()
    product_reviews = get_products_reviwes()
