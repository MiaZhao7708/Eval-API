import requests
from bs4 import BeautifulSoup

def get_RMB_exchange(target_currency='美元'):
    post_url = 'https://srh.bankofchina.com/search/whpj/search_cn.jsp'
    headers={
        'Content-Type': 'application/x-www-form-urlencoded',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Sec-Ch-Ua-Platform': 'macOS',
        'Sec-Fetch-Dest': 'document'
    }
    data = {
        'erectDate': '',
        'nothing': '',
        'pjname': target_currency,
        'head': 'head_620.js',
        'bottom': 'bottom_591.js'
    }
    req = requests.post(url=post_url, headers=headers, data=data)
    html_root = BeautifulSoup(req.text, features="html.parser")
    html_content = html_root.find('div', attrs={'class':'BOC_main publish'})
    table = html_content.find('table')

    tab_headers = [header.text for header in table.find_all('th')]
    res = {}
    for row in table.find_all('tr'):
        tds = row.find_all('td')
        if len(tds) == len(tab_headers):
            for i, key in enumerate(tab_headers):
                res[key] = tds[i].text.strip()
            break
    if '中行折算价' in res.keys():
        return float(res['中行折算价']) / 100
    else:
        print('Parse Currency Error!!! return 1. The main content is:')
        print(html_content)
    return 1