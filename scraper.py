# https://www.trendyol.com/g-skill/ripjawsv-16gb-2x8-ddr4-3600mhz-cl18-siyah-1-35v-f4-3600c18d-16gvk-p-32064675/yorumlar?boutiqueId=61&merchantId=624588
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import time


target = 'https://www.trendyol.com/g-skill/ripjawsv-16gb-2x8-ddr4-3600mhz-cl18-siyah-1-35v-f4-3600c18d-16gvk-p-32064675/yorumlar'


def scroll_down(browser):
    # Sayfanın altına kaydır
    browser.execute_script('window.scrollTo(0, document.body.scrollHeight);')

    # 2 saniye bekle
    time.sleep(2)

    # Sayfanın sonuna ulaşıp ulaşmadığını kontrol et
    scroll_height = browser.execute_script('return document.body.scrollHeight')
    current_height = browser.execute_script('return window.scrollY + window.innerHeight')

    if current_height >= scroll_height:
        print("Sayfanın sonuna ulaşıldı.")
        return True
    else:
        print("Sayfanın sonuna ulaşılamadı.")
        scroll_down(browser)



driver = Chrome()
driver.get(target)

scroll_down(driver)

comment_path = '//*[@id="rating-and-review-app"]/div/div/div/div[3]/div/div/div[3]/div[2]'
# class="comment"
comment_bar = driver.find_element(By.XPATH, comment_path)
counter = 0
with open('comment.txt', 'a+', encoding='utf-8') as cf:
    for comment in comment_bar.find_elements(By.CLASS_NAME, 'comment-text'):
        cf.write(f'{comment.text}\n')
        counter += 1
        print(f'{counter}. {comment.text}')

input('wait')
