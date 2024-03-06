import scrapy


class BooksBasicSpider(scrapy.Spider):
    name = "books_basic"
    allowed_domains = ["books.toscrape.com"]
    start_urls = ["https://books.toscrape.com/catalogue/category/books/fantasy_19/index.html"]

    def parse(self, response):
        books = response.xpath('//h3')
        # books = response.css('h3')
        for book in books:
            yield {
                    'Title': books.xpath('.//a/@title').get(),
                    'URL': books.xpath('.//a/@href').get(),
                    # 'Title': books.css('a::attr(title)').get()
                    # 'URL': books.css('a::attr(href)').get()
            }
