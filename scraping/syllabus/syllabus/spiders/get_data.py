import scrapy


class GetDataSpider(scrapy.Spider):
    name = "get_data"
    allowed_domains = ["student.iimc.kyoto-u.ac.jp"]
    start_urls = ["https://student.iimc.kyoto-u.ac.jp"]

    def parse(self, response):
        pass
