from __future__ import annotations

import runpy


if __name__ == "__main__":
    runpy.run_module("src.data_processing.crawl.topcv_crawler", run_name="__main__")
