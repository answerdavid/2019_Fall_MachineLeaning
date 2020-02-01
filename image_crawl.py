# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'C:\ProgramData\Anaconda3\Lib\site-packages')

from icrawler.builtin import GoogleImageCrawler


def main():
    word = '건물 도장상태 불량'   # search word
    dir_name = './temp1'     # saving directory
    

    google_crawler = GoogleImageCrawler(
        feeder_threads=1,
        parser_threads=1,
        downloader_threads=4,
        storage={'root_dir': dir_name})

    # max_num = number of pictures
    google_crawler.crawl(keyword=word, offset=0, max_num=150,
                         min_size=(200, 200), max_size=None, file_idx_offset=0)


if __name__ == "__main__":
    main()