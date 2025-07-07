from index_jobsposting.config import*
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel, AutoModelForCausalLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import numpy as np
from transformers import pipeline
#from transformers import pipeline
from bs4 import BeautifulSoup
import underthesea
from underthesea import sent_tokenize, word_tokenize
import pickle
import unidecode
import language_tool_python
import torch

# làm sạch ký tự
def clean_text(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text('\n')

# convert unicode
uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"


def loaddicchar():
    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic


dicchar = loaddicchar()


def covert_unicode(txt):
    return re.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)




# kiểm tra điều kiện 1
# vị trí k rõ ràng
def check_dk_1(dict):
    new_city = dict['new_city']
    log = ""
    count = 0
    if new_city != None:
        new_city = new_city.split(',')      
        for city in new_city:
           if int(city) <= 0 or int(city) > 63:
                count = count + 1
                cit_name = dict['cit_name'] 
                if cit_name in TINH_THANH.keys():
                   new_city = TINH_THANH[cit_name][1]   
                else:
                    log += "Tỉnh thành không hợp lệ"
                    count = 1               
    else:
        count = 1
        log += "Không tồn tại tỉnh thành"
    return count, new_city, log


def remove_accent(text):
    return unidecode.unidecode(text)

def remove_quanhuyen_tinhthanh(new_addr, new_name_cit, new_name_qh):
    new_addr = new_addr.lower()
    new_name_cit = new_name_cit.lower()
    new_name_qh = new_name_qh.lower()
    new_name_qh = new_name_qh.replace('thành phố ', 't ')
    new_name_qh = new_name_qh.replace('thị xã ', 'x ')
    new_name_qh = new_name_qh.replace('quận ', 'q ')
    new_name_qh = new_name_qh.replace('huyện ', 'h ')
    addr = new_addr
    addr = addr.replace('tỉnh ', '')
    addr = addr.replace('thành phố ', 't ')
    addr = addr.replace('thị xã ', 'x ')
    addr = addr.replace('quận ', 'q ')
    addr = addr.replace('huyện ', 'h ')
    addr = addr.replace('phường ', 'p ')
    addr = addr.replace('tx.', 'x ')
    addr = addr.replace('tp.', 't ')
    addr = addr.replace('tp', 't ')
    addr = addr.replace('tx', 'x ')
    addr = addr.replace('q.', 'q ')
    addr = addr.replace('p.', 'p ')
    addr = addr.replace('.', ',')
    addr = addr.replace(', ', ',')
    addr = addr.replace('  ', ' ')
    count = 0
    if ',' not in new_name_qh or ',' not in new_name_cit:
        for quan_new in QUAN_NEW.values():
            quan_new = quan_new.lower()
            if quan_new in addr:
                addr = addr.replace(quan_new, list(QUAN_NEW.keys())[list(QUAN_NEW.values()).index(quan_new)])
        if new_name_qh != '' or new_name_cit != '':
            addr = addr.replace('–', ',')
            addr = addr.replace('-', ',')
            addr = addr.replace(' , ', ',')
            addr = addr.replace(' ,', ',')
            addr = addr.replace(', ', ',')
            if ',' in addr:
                addr = addr.replace(',t '+new_name_cit, '')
                addr = addr.replace(','+new_name_cit, '')
                addr = addr.replace(',t hcm', '')
                addr = addr.replace(',t hn', '')
                if 'quận' in addr or 'q ' in addr or 'q.' in addr or 'thành phố' in addr or 'tp' in addr or 'thị xã' in addr or 'tx' in addr or 'huyện' in addr:
                    addr = addr.replace(','+new_name_qh, '')
                else:
                    if len(new_name_qh) > 3:
                        new_name_qh = new_name_qh[2:]
                        addr = addr.replace(','+new_name_qh, '')
                    else: 
                        count = count + 0
                addr = addr.replace(',việt nam,', '')
                addr = addr.replace(',việt nam', '')
            else:
                addr = addr.replace('t '+new_name_cit, '')
                addr = addr.replace(new_name_cit, '')
                addr = addr.replace(',t hcm', '')
                addr = addr.replace(',t hn', '')
                if 'quận' in addr or 'q ' in addr or 'q.' in addr or 'thành phố' in addr or 'tp' in addr or 'thị xã' in addr or 'tx' in addr or 'huyện' in addr:
                    addr = addr.replace(new_name_qh, '')
                else:
                    if len(new_name_qh) > 3:
                        new_name_qh = new_name_qh[2:]
                        addr = addr.replace(new_name_qh, '')
                    else:
                        count = count + 0
                addr = addr.replace('việt nam', '')
                addr = addr.replace(',việt nam', '')
        #if 'lầu' in addr or 'tòa' in addr:
    return addr

# kiểm tra điều kiện 2
# địa chỉ cụ thể không rõ ràng, địa chỉ công ty trùng chỗ làm new_addr == usc_address
def check_dk_2(dict):
    log = ""
    cou = 1
    new_name_qh = dict['new_name_qh']
    #new_name_qh = new_name_qh.split(',')
    usc_address = dict['usc_address']
    new_name_cit = dict['new_name_cit']
    mota = dict['new_mota']
    title = dict['new_title']
    cit_name = dict['cit_name']
    count = 0
    back_addr = ""
    # addr = new_addr
    if usc_address.lower() == 'toàn quốc' or usc_address.lower() == 'tuyển dụng toàn quốc':
        log += f'{cou}, Bản tin không được phép có địa chỉ ở toàn quốc\n'
        count = 1
        cou = cou + 1
    if 'toàn quốc' in cit_name.lower() or 'tuyển dụng toàn quốc' in cit_name.lower():
        log += f"{cou}, Tỉnh thành phải cụ thể, không được là toàn quốc\n"
        count = 1
        cou = cou + 1
    if 'toàn quốc' in usc_address:
        usc_address = re.sub('toàn quốc', '', usc_address, flags=re.IGNORECASE)
        usc_address = re.sub(r'\s{2,}', ' ', usc_address).strip()
    if 'tuyển dụng toàn quốc' in usc_address:
        usc_address = re.sub('tuyển dụng toàn quốc', '', usc_address, flags=re.IGNORECASE)
        usc_address = re.sub(r'\s{2,}', ' ', usc_address).strip()  
    if 'toàn quốc' in title:
        title = re.sub('toàn quốc', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\s{2,}', ' ', title).strip()
    if 'tuyển dụng toàn quốc' in title:
        title = re.sub('tuyển dụng toàn quốc', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\s{2,}', ' ', title).strip() 
    if 'toàn quốc' in mota:
        mota = re.sub('toàn quốc', '', mota, flags=re.IGNORECASE)
        mota = re.sub(r'\s{2,}', ' ', mota).strip()
    if 'tuyển dụng toàn quốc' in mota:
        mota = re.sub('tuyển dụng toàn quốc', '', mota, flags=re.IGNORECASE)
        mota = re.sub(r'\s{2,}', ' ', mota).strip() 
    back_addr = usc_address
    usc_address = usc_address.lower()
    usc_address = remove_quanhuyen_tinhthanh(usc_address, new_name_cit, new_name_qh)
    addr = usc_address
    usc_address = usc_address.split(',')
    print('usc_address:', usc_address)
    if 'tòa' not in usc_address[0] and 'khu' not in usc_address[0] and 'kcn' not in usc_address[0] and 'cụm' not in usc_address[0] and 'ccn' not in usc_address[0] and 'lô' not in usc_address[0] and 'chung cư' not in usc_address[0] and 'plaza' not in usc_address[0] and 'tầng' not in usc_address[0] and 'lầu' not in usc_address[0] and 'số' not in usc_address[0]: 
        print('tessst:', usc_address[0])
        if len(usc_address)>1:
            if re.search("(\d+\w*)", usc_address[0]) == None and 'số' not in usc_address[1] and re.search("(\d+\w*)", usc_address[1]) == None and len(usc_address)<4:
                log += f'{cou}, Địa chỉ không chi tiết\n'
                count = 1
                cou = cou + 1
        else:
            patterns = ['phường [1-9]{1}', 'phường [0-9]{2}', 'p [1-9]{1}', 'p [0-9]{2}', 'p.[1-9]{1}', 'p.[0-9]{2}',
                        'quận [1-9]{1}', 'quận [0-9]{2}', 'q [1-9]{1}', 'q [0-9]{2}', 'q.[1-9]{1}', 'q.[0-9]{2}']
            for pattern in patterns:
                new_add = usc_address[0].replace(pattern, ' ')
            if re.search("(\d+\w*)", new_add) == None:
                print('usc_address:', new_add)
                count = 1
                log +=  f'{cou}, Địa chỉ không chi tiết'
                cou = cou + 1
    else:
        if (('tầng' in addr or 'lầu' in addr) and ('tầng' not in usc_address and 'lầu' not in usc_address)) or (('tầng' in usc_address or 'lầu' in usc_address) and ('tầng' not in addr and 'lầu' not in addr)):
            print('1 trong 2 có tầng')
            pat = ['tầng [0-9]{2},', 'lầu [0-9]{2},', 'lầu [0-9]{1},', 'tầng [0-9]{1},', 'lầu [0-9]{2}', 'tầng [0-9]{2}', 'lầu [0-9]{1}', 'tầng [0-9]{1}']
            for pa in pat:
                if re.findall(pa, addr):
                    addr = addr.replace(re.findall(pa, addr)[0], '')
#                if re.findall(pa, usc_address):
 #                   usc_address = usc_address.replace(re.findall(pa, usc_address)[0], '')   
    if addr != '':
        if addr != remove_accent(addr):
            addr = addr.lower()
            reps = ['thành phố', 'quận', 'phường', 'q.', 'p.', 'huyện', 'tp', '-', '–', 'xã', 'thị trấn', 'thị xã', 'ấp', 'khu phố', 'tỉnh', 'thôn', '.', 'số', 'số nhà', 'tòa', 'toà', 'tòa nhà', 'toà nhà', 'tầng', 'lầu', 'đường', 'tổ']
            addr = addr.replace('hồ chí minh', 'hcm')
            addr = addr.replace('hồ chí minh', 'hcm')
            for rep in reps:
                addr = addr.replace(rep, '')
                addr = addr.replace(rep, '')
            addr = addr.replace(',', ' ')
            addr = addr.replace(',', ' ')
            print('tessssst:', usc_address)
            print('tes:', addr)
            addr = addr.split()
            print('tessssst:', addr)
            if 'và' in addr or 'hoặc' in addr:
                count = count + 1
                log += f'{cou}, Có nhiều địa chỉ chi tiết\n'
                count = 1
                cou = cou + 1
        else:
            log +=  f'{cou}, Không có địa chỉ công ty hoặc địa chỉ không đầy đủ dấu câu\n'
            count = 1
            cou = cou + 1
    else:
        log += f'{cou}, Không có địa chỉ công ty hoặc địa chỉ không đầy đủ dấu câu\n'
        count = 1
        cou = cou + 1
    #else:
    #    addr = remove_accent(new_addr)
    #    reps = ['thanh pho', 'quan', 'q ', 'phuong', 'p ', 'q.', 'p.', 'huyen', 'tp', ',', ' ']
    return count, back_addr, title, mota, log
# kiểm tra điều kiện 3
# hình thức làm việc phải là nhân viên chính thức
def check_dk_3(dict):
    log = ""
    #new_hinhthuc = dict['new_hinh_thuc']
    new_hinh_thuc_text = dict['new_hinh_thuc_text']
    new_hinh_thuc_text = "".join(ch for ch in new_hinh_thuc_text.lower() if ch.isalnum() or ch.isspace())
    count = 0
    #if new_hinhthuc != str(1) or new_hinh_thuc_text != "Toàn thời gian cố định":
    pattern = ['bán thời gian', 'bán thời gian tạm thời', 'việc làm từ xa', 'parttime', 'học việc']
    for pat in pattern:
        if pat in new_hinh_thuc_text:
            count = 1
            log = "Chỉ chấp nhận toàn thời gian\n"
            print(log)
            break
    if log == "":
        print("Không tìm thấy parttime " + new_hinh_thuc_text)
    return count, log

# kiểm tra điều kiện 4
# Lương không được tính theo giờ
def check_dk_4(dict):
    log = ""
    new_mota = dict['new_mota']
    new_mota = clean_text(new_mota).lower()
    new_quyenloi = dict['new_quyenloi']
    new_quyenloi = clean_text(new_quyenloi).lower()
    new_title = dict['new_title'].lower()
    new_money_str = dict['new_money_str']
    count = 0
    new_mota = new_mota.replace(" ", "")
    new_quyenloi = new_mota.replace(" ", "")
    new_money_str = new_money_str.replace(" ", "")
    pattern_1 = '[0-9]{2}k/h'
    pattern_2 = '[0-9]{2}k/giờ'
    if pattern_1 in new_mota.lower() or pattern_2 in new_mota.lower():
       count = 1
       log = "Lương không được tính theo giờ"
    if pattern_1 in new_quyenloi.lower() or pattern_2 in new_quyenloi.lower() :
       count = 1
       log = "Lương không được tính theo giờ"
#    if pattern_1 in new_title.lower() or pattern_2 in new_title.lower():
#       count = 1
#       log = "Lương không được có trong tiêu đề. --Đã loại bỏ--"
#       new_title = re.sub(pattern_1, "", new_title)
    if pattern_1 in new_money_str.lower() or pattern_2 in new_money_str.lower():
       count = 1
       log = "Lương không được tính theo giờ"


    return count, log

# check địa chỉ
def extract_address(text):
    address = []
    tokenizer = AutoTokenizer.from_pretrained("NlpHUST/ner-vietnamese-electra-base", model_max_length=50)
    model = AutoModelForTokenClassification.from_pretrained("NlpHUST/ner-vietnamese-electra-base")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="max")
    ner_results = nlp(text)
    for ent in ner_results:
        if (ent['entity_group'] == 'LOCATION'):
            address.append(ent['word'])
    return address

# check lương 2 đầu
def check_salary(new_money_str, text_salary):
    money = ['0']
    text_money = ['0']
    count = 0
    patterns = PATTERN_SALARY
    new_money_str = new_money_str.replace(' ', '')
    new_money_str = new_money_str.replace('USD', 'usd')
    #luongs = ['Lương', 'thu nhập', 'lương', 'Thu nhập']
    for pattern in patterns:
        if (re.findall(pattern, new_money_str)):
            money = re.findall(pattern, new_money_str)
            print('mo:', money)
    for text in text_salary:
        text = text.replace(' ', '')
        text = text.replace('–', '-')
        text = text.replace('->', '-')
        text = text.replace('~', '-')
        for pattern in patterns:
            #for luong in luongs:
            if (re.findall(pattern, text)): # and (re.findall(luong, text)):
                text_money = re.findall(pattern, text)
                print('pa:', text_money)
    text_money[0] = text_money[0].replace('đến', '-')
    money[0] = money[0].lower()
    money[0] = money[0].replace('triệu', '')
    money[0] = money[0].replace('tr', '')
    money[0] = money[0].replace('.000usd', '000')
    money[0] = money[0].replace('usd', '')
    money[0] = money[0].replace(' ', '')
    if 'triệu' in text_money[0] or 'tr' in text_money[0] or 'm' in text_money[0]:
        text_money[0] = text_money[0].replace(',', '.')
        text_money[0] = text_money[0].replace('triệu', '')
        text_money[0] = text_money[0].replace('tr', '')
        text_money[0] = text_money[0].replace('m', '')
        text_money[0] = text_money[0].replace('.000usd', '000')
        text_money[0] = text_money[0].replace('usd', '')
        text_money[0] = text_money[0].replace(' ', '')
        if (money[0] != text_money[0] and text_money[0] != '0'):
            count = count + 1
            print('money:', money[0])
            print('money_ql:', text_money[0])
            print('th1 của lương 2 đầu')
    else:
        text_money[0] = text_money[0].replace(',', '.')
        text_money[0] = text_money[0].replace('.000.000', '')
        text_money[0] = text_money[0].replace('00.000', '')
        text_money[0] = text_money[0].replace('0.000', '')
        text_money[0] = text_money[0].replace('đến', '-')
        text_money[0] = text_money[0].replace('vnđ', '')
        text_money[0] = text_money[0].replace('vnd', '')
        text_money[0] = text_money[0].replace('đ', '')
        text_money[0] = text_money[0].replace('.000usd', '000')
        text_money[0] = text_money[0].replace('usd', '')
        if money[0] != text_money[0] and text_money[0] != '0':
            count = count + 1
            print('money:', (money[0]))
            print('money_ql:', text_money[0])
            print('th2 của lương 2 đầu')
    return count, text_money, money

# kiểm tra lương min hoặc max
def check_min_max(new_money_str, text_salary):
    money = ['0']
    text_money = ['0']
    count = 0
    new_money_str = new_money_str.replace(' ', '')
    new_money_str = new_money_str.replace('USD', 'usd')
    # check lương min hoặc lương max
    patterns = PATTERN_MIN_MAX
    for pattern in patterns:
        if (re.findall(pattern, new_money_str)):
            money = re.findall(pattern, new_money_str)
    for text in text_salary:
        text = text.replace(' ', '')
        text = text.replace('–', '-')
        text = text.replace('->', '-')
        text = text.replace('~', '-')
        for pattern in PATTERN_MIN_MAX:
            if (re.findall(pattern, text)):
                for luong in re.findall(pattern, text):
                    print('l:', text)
                    if len(re.findall('[0-9]{10}', luong)) == 0 and len(re.findall('[0-9]{9}', luong)) == 0 and len(re.findall('[0-1]{1}.[0-9]{1}m', luong)) == 0 and len(re.findall('[0-1]{1}.[0-9]{2}m', luong)) == 0 and '-' not in luong:
                        text_money = re.findall(pattern, text)
                        print('money:', text_money)
                        print('pa:', pattern)
    money[0] = money[0].lower()
    money[0] = money[0].replace('triệu', '')
    money[0] = money[0].replace('tr', '')
    money[0] = money[0].replace('.000usd', '000')
    money[0] = money[0].replace('usd', '')
    money[0] = money[0].replace(' ', '')
    if 'tr' in text_money[0] or 'm' in text_money[0]:
        text_money[0] = text_money[0].replace(',', '.')
        text_money[0] = text_money[0].replace('triệu', '')
        text_money[0] = text_money[0].replace('tr', '')
        text_money[0] = text_money[0].replace('m', '')
        text_money[0] = text_money[0].replace('.000usd', '000')
        text_money[0] = text_money[0].replace('usd', '')
        text_money[0] = text_money[0].replace(' ', '')
        if (money[0] != text_money[0] and text_money[0] != '0'):
            count = count + 1
            print('money_max:', (money[0]))
            print('money_ql_max:', text_money[0])
            print('th1 của lương 1 đầu')
    else:
        text_money[0] = text_money[0].replace(',', '.')
        text_money[0] = text_money[0].replace('.000.000', '')
        text_money[0] = text_money[0].replace('00.000', '')
        text_money[0] = text_money[0].replace('0.000', '')
        text_money[0] = text_money[0].replace('usd', '')
        text_money[0] = text_money[0].replace(' ', '')
        if money[0] != text_money[0] and text_money[0] != '0':
            count = count + 1
            print('money:', (money[0]))
            print('money_ql:', text_money[0])
            print('th2 của lương 1 đầu')
    return count, text_money, money

# kiểm tra lương hỗn hợp: money là lương 2 đầu nhưng lương mota, quyền lợi là lương 1 đầu

def check_mix(money, mon):
    count = 0
    # money là lương 2 đầu
    if 'tr' in mon[0]:
        mon[0] = mon[0].replace(',', '.')
        mon[0] = mon[0].replace('triệu', '')
        mon[0] = mon[0].replace('tr', '')
        mon[0] = mon[0].replace(' ', '')
        if (mon[0] not in money[0] and mon[0] != '0'):
            count = count + 1
            print('money:', (money[0]))
            print('money_ql:', mon[0])
    else:
        mon[0] = mon[0].replace(',', '.')
        mon[0] = mon[0].replace('.000.000', '')
        mon[0] = mon[0].replace('00.000', '')
        mon[0] = mon[0].replace('0.000', '')
        mon[0] = mon[0].replace(' ', '')
        if mon[0] not in money[0] and mon[0] != '0':
            count = count + 1
            print('money:', (money[0]))
            print('money_ql:', mon[0])
        if mon[0] == '':
            count = count + 1
    return count

def mix_check(money, mon):
    # money là lương 1 đầu
    count = 0
    if 'tr' in mon[0]:
        mon[0] = mon[0].replace(',', '.')
        mon[0] = mon[0].replace('triệu', '')
        mon[0] = mon[0].replace('tr', '')
        mon[0] = mon[0].replace(' ', '')
        if (money[0] not in mon[0] and mon[0] != '0'):
            count = count + 1
            print('money:', (money[0]))
            print('money_ql:', mon[0])
    else:
        mon[0] = mon[0].replace(',', '.')
        mon[0] = mon[0].replace('.000.000', '')
        mon[0] = mon[0].replace('00.000', '')
        mon[0] = mon[0].replace('0.000', '')
        mon[0] = mon[0].replace(' ', '')
        if money[0] not in mon[0] and mon[0] != '0':
            count = count + 1
            print('money:', (money[0]))
            print('money_ql:', mon[0])
    return count

def check_number_salary(texts, pattern_two, pattern_one):
    dem = 0
    print(dem)
    for text in texts:
        for pattern_t in pattern_two:
            text = text.replace('–', '-')
            text = text.replace('->', '-')
            text = text.replace(' ', '')
            if (re.findall(pattern_t, text)):
                dem = dem + 1
                print(re.findall(pattern_t, text))
                print('text_check:', text)
                text = text.replace(re.findall(pattern_t, text)[0], '')
        print('check_text:', text)
        for pattern_o in pattern_one:
            text = text.replace(' ', '')
            if (re.findall(pattern_o, text)) and len(re.findall('[0-9]{9}', text)) == 0 and len(re.findall('[0-1]{1}.[0-9]{1}m', text)) == 0 and len(re.findall('[0-1]{1}.[0-9]{2}m', text)) == 0:
                print('nhieuluong:', re.findall(pattern_o, text))
                dem = dem + 1
                print(dem)
    return dem


# kiểm tra điều kiện 5 jobsposting
# Chỉ được phép co 1 mức lương và địa chỉ phải khớp nhau, lương 2 đầu
def check_dk_5(dict):
    log = ""
    cou = 1
    new_mota = dict['new_mota']
    new_mota = clean_text(new_mota)
    new_mota = new_mota.replace('\n\xa0', ' ')
    new_mota = new_mota.replace('\n', ' ')
    new_mota = new_mota.replace('$', 'usd')
    new_mota = new_mota.lower()
    new_quyenloi = dict['new_quyenloi']
    new_quyenloi = clean_text(new_quyenloi)
    new_quyenloi = new_quyenloi.replace('\n\xa0', ' ')
    new_quyenloi = new_quyenloi.replace('\n', ' ')
    new_quyenloi = new_quyenloi.replace('$', 'usd')
    new_quyenloi = new_quyenloi.lower()
    new_yeucau = dict['new_yeucau']
    new_yeucau = clean_text(new_yeucau)
    new_yeucau = new_yeucau.replace('\n\xa0', ' ')
    new_yeucau = new_yeucau.replace('\n', ' ')
    new_yeucau = new_yeucau.replace('$', 'usd')
    new_yeucau = new_yeucau.lower()
    new_name_cit = dict['new_name_cit']
    new_name_cit = new_name_cit.split(',')
    count = 0

    # kiểm tra trùng lặp địa chỉ
    huyen_check = 0
    tinh_check = 0
    check_add = new_mota + new_quyenloi + new_yeucau
    check_add = check_add.replace('.', ' ')
    check_add = underthesea.word_tokenize(check_add, fixed_words=FIXED_WORDS, format="text")
    check_add = check_add.split()
    if 'địa_điểm_làm_việc' in check_add or 'địa_chỉ_làm_việc' in check_add or 'nơi_làm_việc' in check_add:
        for quanhuyen in QUAN_FIXED:
            if quanhuyen in check_add:
                huyen_check = huyen_check + 1
                check_add = check_add.replace(quanhuyen, '')
        for tinhthanh in TINH_FIXED:
            if tinhthanh in check_add:
                tinh_check = tinh_check + 1
                check_add = check_add.replace(tinhthanh, '')
    if tinh_check > 1 or huyen_check > 1:
        count = count + 1
        log += f'{cou}, Có nhiều địa chỉ trong nội dung\n'
        cou = cou + 1

    yeucau = new_yeucau.split('\r')
    quyenloi = new_quyenloi.split('\r')
    mota = new_mota.split('\r')
    for ql in quyenloi:
        ql = ql.split()
        for q in ql:
            q = q.replace('(', ' ')
            q = q.replace(')', ' ')
            q = q.replace('.', ' ')
            q = q.replace(',', ' ')
            q = q.replace(':', ' ')
            if 'tr' in q and q != 'tr' and q != 'triệu'  and q != 'triệu/tháng' and q != 'tr/tháng' and q != 'tr/' and q != 'triệu/' and re.compile('\d').search(q) == None:
                new_quyenloi = new_quyenloi.replace(q, '')
            if 'm' in q and q != 'm' and re.compile('\d').search(q) == None:
                new_quyenloi = new_quyenloi.replace(q, '')
    for mt in mota:
        mt = mt.split()
        for m in mt:
            m = m.replace('(', ' ')
            m = m.replace(')', ' ')
            m = m.replace('.', ' ')
            m = m.replace(',', ' ')
            m = m.replace(':', ' ')
            if 'tr' in m and m != 'tr' and m != 'triệu' and m != 'triệu/tháng' and m != 'tr/tháng' and m != 'tr/' and m != 'triệu/' and re.compile('\d').search(m) == None:
                new_mota = new_mota.replace(m, '')
            if 'm' in m and m != 'm' and re.compile('\d').search(m) == None:
                new_mota = new_mota.replace(m, '')
    for yc in yeucau:
        yc = yc.split()
        for y in yc:
            y = y.replace('(', ' ')
            y = y.replace(')', ' ')
            y = y.replace('.', ' ')
            y = y.replace(',', ' ')
            y = y.replace(':', ' ')
            if 'tr' in y and y != 'tr' and y != 'triệu' and y != 'triệu/tháng' and y != 'tr/tháng' and y != 'tr/' and y != 'triệu/' and re.compile('\d').search(y) == None:
                new_yeucau = new_yeucau.replace(y, '')
            if 'm' in y and y != 'm' and re.compile('\d').search(y) == None:
                new_yeucau = new_yeucau.replace(y, '')
    new_quyenloi = new_quyenloi.split('\r')
    new_yeucau = new_yeucau.split('\r')
    motas = new_mota.split('\r')

    print('new_quyenloi:', new_quyenloi)
    print('new_yeucau:', new_yeucau)
    print('motas:', motas)
    # kiểm tra trùng lặp mức lương
    # check lương 2 đầu mút
    new_money_str = dict['new_money_str']
    print('test:', new_money_str)
    min_ql = 0
    min_mt = 0 
    min_yc = 0
    count_ql, money_ql, money = check_salary(new_money_str, new_quyenloi)
    count_mt, money_mt, money = check_salary(new_money_str, motas)
    count_yc, money_yc, money = check_salary(new_money_str, new_yeucau)
    min_ql, mon_ql, mon = check_min_max(new_money_str, new_quyenloi)
    min_mt, mon_mt, mon = check_min_max(new_money_str, motas)
    min_yc, mon_yc, mon = check_min_max(new_money_str, new_yeucau)
    if money[0] == '0':
        if new_money_str == 'Thỏa thuận':
            if mon_ql[0] == '0' and mon_mt[0] == '0' and mon_yc[0] == '0':
                count = count + 0
            else:
                log += f'{cou}, Thỏa thuận nhưng vẫn nhắc đến lương\n'
                count = count + 1
                cou =  cou + 1
        else:
            count = count + 0
 #           log +=  f'{cou}, Lương 1 đầu trong money str\n'
 #           cou = cou + 1
    else:
        if (money_ql[0] == '0' and money_mt[0] == '0' and money_yc[0] == '0'):
            if (mon_ql[0] != '0' or mon_mt[0] != '0' or mon_yc[0] != '0'):
                count = count + 0
     #           print('khong tim duoc luong 2 dau nhung co luong 1 dau')
            else:
                if new_money_str == "" or "tr" not in new_money_str.lower():
                   log += f'{cou}, Không có lương trong nội dung\n'
                   count = count + 1
                   cou = cou + 1
        else:
            count = 0
    check_mota = check_number_salary(motas, CHECK_SALARY_TWO, CHECK_SALARY_ONE)
    check_yeucau = check_number_salary(new_yeucau, CHECK_SALARY_TWO, CHECK_SALARY_ONE)
    check_quyenloi = check_number_salary(new_quyenloi, CHECK_SALARY_TWO, CHECK_SALARY_ONE)
    if (check_mota + check_yeucau + check_quyenloi) > 1:
        count = count + 1
        log += f'{cou}, Có nhiều lương trong nội dung\n'
        cou = cou + 1
    return count, log


# kiểm tra điều kiện 5 index
# Chỉ được phép co 1 mức lương và địa chỉ phải khớp nhau
def check_dk_5_1(dict):
    new_mota = dict['new_mota']
    new_mota = clean_text(new_mota)
    new_mota = new_mota.replace('\n\xa0', ' ')
    new_mota = new_mota.replace('\xa0\n', ' ')
    #new_mota = new_mota.replace('\n', ' ')
    new_mota = new_mota.replace('$', 'usd')
    new_mota = new_mota.lower()
    
    new_quyenloi = dict['new_quyenloi']
    new_quyenloi = clean_text(new_quyenloi)
    new_quyenloi = new_quyenloi.replace('\n\xa0', ' ')
    new_quyenloi = new_quyenloi.replace('\xa0\n', ' ')
    #new_quyenloi = new_quyenloi.replace('\n', '\r')
    new_quyenloi = new_quyenloi.replace('$', 'usd')
    new_quyenloi = new_quyenloi.lower()
    
    new_yeucau = dict['new_yeucau']
    new_yeucau = clean_text(new_yeucau)
    new_yeucau = new_yeucau.replace('\n\xa0', ' ')
    new_yeucau = new_yeucau.replace('\xa0\n', ' ')
    #new_yeucau = new_yeucau.replace('\n', '\r')
    new_yeucau = new_yeucau.replace('$', 'usd')
    new_yeucau = new_yeucau.lower()

    new_name_cit = dict['new_name_cit']
    new_name_cit = new_name_cit.split(',')
    count = 0
    # kiểm tra trùng lặp địa chỉ
    huyen_check = 0
    tinh_check = 0
    # check_add = new_mota + new_quyenloi + new_yeucau
    # check_add = check_add.replace('.', ' ')
    # #check_add = underthesea.word_tokenize(check_add, fixed_words=FIXED_WORDS, format="text")
    # check_addr = check_add.split('\n')
    # print('check_addr:', check_addr)
    # for diachi in check_addr:
    #     print('diachi:', diachi)
    #     if 'địa_điểm_làm_việc' in diachi or 'địa_chỉ_làm_việc' in diachi or 'nơi_làm_việc' in diachi:
    #         for quanhuyen in QUAN_FIXED:
    #             if quanhuyen in diachi:
    #                 huyen_check = huyen_check + 1
    #                 print('huyen:', quanhuyen)
    #         for tinhthanh in TINH_FIXED:
    #             if tinhthanh in diachi:
    #                 tinh_check = tinh_check + 1
    #                 print('tinh:', tinhthanh)
    # if tinh_check > 1 or huyen_check > 1:
    #     count = count + 1
    #     print('có nhiều địa chỉ trong nội dung')
    yeucau = new_yeucau.split('\r\n')
    quyenloi = new_quyenloi.split('\r\n')
    mota = new_mota.split('\r\n')
    for ql in quyenloi:
        ql = ql.split()
        for q in ql:
            q = q.replace('(', ' ')
            q = q.replace(')', ' ')
            q = q.replace('.', ' ')
            q = q.replace(',', ' ')
            q = q.replace(':', ' ')
            if 'tr' in q and q != 'tr' and q != 'triệu' and q != 'triệu/tháng' and q != 'tr/tháng' and q != 'tr/' and q != 'triệu/' and re.compile('\d').search(q) == None:
                new_quyenloi = new_quyenloi.replace(q, 'chợ')
            if 'm' in q and q != 'm' and re.compile('\d').search(q) == None:
                new_quyenloi = new_quyenloi.replace(q, '')
    for mt in mota:
        mt = mt.split()
        for m in mt:
            m = m.replace('(', ' ')
            m = m.replace(')', ' ')
            m = m.replace('.', ' ')
            m = m.replace(',', ' ')
            m = m.replace(':', ' ')
            if 'tr' in m and m != 'tr' and m != 'triệu' and m != 'triệu/tháng' and m != 'tr/tháng' and m != 'tr/' and m != 'triệu/' and re.compile('\d').search(m) == None:
                new_mota = new_mota.replace(m, 'chợ')
            if 'm' in m and m != 'm' and re.compile('\d').search(m) == None:
                new_mota = new_mota.replace(m, '')
    for yc in yeucau:
        yc = yc.split()
        for y in yc:
            y = y.replace('(', ' ')
            y = y.replace(')', ' ')
            y = y.replace('.', ' ')
            y = y.replace(',', ' ')
            y = y.replace(':', ' ')
            if 'tr' in y and y != 'tr' and y != 'triệu' and y != 'triệu/tháng' and y != 'tr/tháng' and y != 'tr/' and y != 'triệu/' and re.compile('\d').search(y) == None:
                new_yeucau = new_yeucau.replace(y, 'chợ')
            if 'm' in y and y != 'm' and re.compile('\d').search(y) == None:
                new_yeucau = new_yeucau.replace(y, '')
                new_yeucau = new_yeucau.replace('members', '')
    new_quyenloi = new_quyenloi.split('\r\n')
    new_yeucau = new_yeucau.split('\r\n')
    motas = new_mota.split('\r\n')

    # print('new_quyenloi:', new_quyenloi)
    # print('new_yeucau:', new_yeucau)
    # print('motas:', motas)
    # kiểm tra trùng lặp mức lương
    # check lương 2 đầu mút
    new_money_str = dict['new_money_str']
    print('test:', new_money_str)
    count_ql, money_ql, money = check_salary(new_money_str, new_quyenloi)
    count_mt, money_mt, money = check_salary(new_money_str, motas)
    count_yc, money_yc, money = check_salary(new_money_str, new_yeucau)
    # check min max
    min_ql = 0
    min_mt = 0 
    min_yc = 0
    min_ql, mon_ql, mon = check_min_max(new_money_str, new_quyenloi)
    min_mt, mon_mt, mon = check_min_max(new_money_str, motas)
    min_yc, mon_yc, mon = check_min_max(new_money_str, new_yeucau)
    if money[0] != '0':
        #nếu không tìm được lương 2 đầu trong nội dung
        if money_ql[0] == '0' and money_mt[0] == '0' and money_yc[0] == '0':
            # nhưng có lương 1 đầu trong nội dung
            if (mon_ql[0] != '0' or mon_mt[0] != '0' or mon_yc[0] != '0'):
                count = count + 1
                print('lương 1 đầu trong nội dung và lương 2 đầu trong money')
                '''
                cou_ql = check_mix(money, mon_ql)
                cou_yc = check_mix(money, mon_yc)
                cou_mt = check_mix(money, mon_mt)
                count = count + cou_mt + cou_yc + cou_ql
                '''
            # nếu không có lương trong nội dung
            else:
                print('không có lương trong nội dung và có lương 2 đầu trong money')
                count = count + 0
        # nếu tìm được lương 2 đầu trong nội dung
        else:
            count = count + count_yc + count_ql + count_mt 
            print('nội dung và money chứa lương 2 đầu:', count)
    # nếu money str không phải lương 2 đầu
    else:
        # thì nó có thể là thỏa thuận
        if new_money_str == 'Thỏa thuận':
            # nội dung không chứa lương
            if mon_ql[0] == '0' and mon_mt[0] == '0' and mon_yc[0] == '0':
                    print('money là thỏa thuận và nội dung k có lương')
                    count = count + 0
            else:
                print('thỏa thuận nhưng vẫn nhắc đến lương')
                count = count + 1
        # nếu không phải thỏa thuận => lương 1 đầu
        else:
            # nếu nội dung không chứa lương 2 đầu
            if money_ql[0] == '0' and money_mt[0] == '0' and money_yc[0] == '0':
                #thì có thể chứa lương 1 đầu
                if (mon_ql[0] != '0' or mon_mt[0] != '0' or mon_yc[0] != '0'):
                    print('lương 1 đầu trong nội dung và money')
                    count = count + min_ql + min_mt + min_yc
                else:
                    print('nội dung k có lương, money là lương 1 đầu')
                    count = count + 0
            else:
                print('nội dung chứa lương 2 đầu, money chứa lương 1 đầu')
                count = count + 1
                '''
                ql_cou = mix_check(mon, money_ql)
                yc_cou = mix_check(mon, money_yc)
                mt_cou = mix_check(mon, money_mt)
                count = count + ql_cou + yc_cou + mt_cou
                '''
    check_mota = check_number_salary(motas, CHECK_SALARY_TWO, CHECK_SALARY_ONE)
    check_yeucau = check_number_salary(new_yeucau, CHECK_SALARY_TWO, CHECK_SALARY_ONE)
    check_quyenloi = check_number_salary(new_quyenloi, CHECK_SALARY_TWO, CHECK_SALARY_ONE)
    if (check_mota + check_yeucau + check_quyenloi) > 1:
        count = count + 1
        print('có nhiều lương trong nội dung')

    return count 


    # kiểm tra điều kiện 6 jonposting
# quận huyện, tỉnh thành ở mục quận huyện và mục địa chỉ phải khớp nhau
def check_dk_6(dict):
    log = ""
    cou = 1
    new_addr = dict['usc_address']
    back_qh = ""
    print(new_addr)
    if 'tp' in new_addr:
        print(11111)
    new_name_cit = name_cit = dict['cit_name']
    new_name_qh = dict['new_qh_id']
    new_city = dict['new_city']
    new_name_cit = new_name_cit.split(',')
    new_name_qh = new_name_qh.lower()
    count = 0
    if ',' in new_name_qh:
        count = count + 1
    else:
        new_name_qh = new_name_qh.replace('thành phố ', 't ')
        new_name_qh = new_name_qh.replace('thị xã ', 'x ')
        new_name_qh = new_name_qh.replace('quận ', 'q ')
        new_name_qh = new_name_qh.replace('huyện ', 'h ')
    print('new_name_qh:', new_name_qh)
    addr = new_addr.lower()
    addr = addr.replace(', Việt Nam', '')
    addr = addr.replace('tỉnh ', '')
    addr = addr.replace('thành phố ', 't ')
    addr = addr.replace('thị xã ', 'x ')
    addr = addr.replace('quận ', 'q ')
    addr = addr.replace('huyện ', 'h ')
    addr = addr.replace('tx.', 'x ')
    addr = addr.replace('tp.', 't ')
    addr = addr.replace('tp', 't ')
    addr = addr.replace('tx', 'x ')
    addr = addr.replace('q.', 'q ')
    addr = addr.replace('.', ',')
    addr = addr.replace(', ', ',')
    addr = addr.replace('  ', ' ')
    addr = addr.replace('toà', 'tòa')
    addr = addr.replace('thủy', 'thuỷ')
    for quan_new in QUAN_NEW.values():
        quan_new = quan_new.lower()
        if quan_new in addr:
            addr = addr.replace(quan_new, list(QUAN_NEW.keys())[list(QUAN_NEW.values()).index(quan_new)])
            new_addr = new_addr.replace(quan_new, list(QUAN_NEW.keys())[list(QUAN_NEW.values()).index(quan_new)])
    print('addr:', addr)
    if new_name_qh == '' or new_name_cit[0] == '':
            count = count + 1
            log += f'{cou}, Không có quận huyện, không có tỉnh thành\n'
    else:
        addr = addr.replace('–', ',')
        addr = addr.replace('-', ',')
        addr = addr.replace(' , ', ',')
        addr = addr.replace(' ,', ',')
        addr = addr.replace(', ', ',')
        if ',' in addr: 
            addr = addr.split(',')
            print('addr1:', addr)
            # if 'tp ' in addr or 'tx ' in addr or 'q ' in addr or :
            tinh_addr = 0
            ct_addr = 0
            for tinh_thanh in TINH_THANH.keys():
                if tinh_thanh.lower() == addr[-1]:
                    tinh_addr = tinh_addr + 1
                    print('tinhthanh:', tinh_thanh)
            if tinh_addr >= 1:
                print('có tỉnh')
                for cit in new_name_cit:
                    for ct in TINH_THANH[cit][0]:
                        print('ct:', ct)
                        if ct.lower() == addr[-1]:
                            ct_addr = ct_addr + 1
                if ct_addr >= 1:
                    print('tỉnh trùng khớp')
                    if 'quận' in new_addr or 'q ' in new_addr or 'q.' in new_addr or 'thành phố' in new_addr or 'tp' in new_addr or 'thị xã' in new_addr or 'tx' in new_addr or 'huyện' in new_addr:
                        print(new_name_qh)
                        if new_name_qh in addr or new_name_qh[2:] in addr:
                            pass
                        else:
                            print(new_name_qh)
                            print('addr:', addr)
                            count = count + 1
                            log += f'{cou}, Không có quận huyện mục quận huyện trong địa chỉ chi tiết\n'
                            cou = cou + 1
                            back_qh = addr[-2]
                    else:
                        print(new_name_qh)
                        if new_name_qh in addr or new_name_qh[2:] in addr:
                            pass
                        else:
                            print(new_name_qh)
                            print('addr:', addr)
                            count = count + 1
                            log += f'{cou}, Không có quận huyện mục quận huyện trong địa chỉ chi tiết\n'
                            cou = cou + 1
                            back_qh = addr[-2]
                else:
                    log += f'{cou}, Tỉnh không khớp\n'
                    cou = cou + 1
                    # vi phạm
                    count = count + 1
            else:
                for tinh_thanh in TINH_THANH.keys():
                   if tinh_thanh.lower() in name_cit.lower():
                      tinh_addr = tinh_addr + 1
                      print('tinhthanh:', tinh_thanh)
                if tinh_addr >= 1:
                    print('có tỉnh')
                else:
                    log += f'{cou}, Không có tỉnh thành\n\n'
                # nếu trong addr không có tỉnh thì xét các huyện trong tỉnh có trong addr không
                new_addr = new_addr.lower()
                for cit in new_name_cit:
                    for quanhuyen, tinhthanh in QUAN_HUYEN_TINH_THANH.items():
                        quanhuyen = quanhuyen.lower()
                        if tinhthanh == cit:
                            if 'quận' in new_addr or 'q ' in new_addr or 'q.' in new_addr or 'thành phố' in new_addr or 'tp' in new_addr or 'thị xã' in new_addr or 'tx' in new_addr or 'huyện' in new_addr:
                                quanhuyen = quanhuyen.replace('thành phố ', 't ')
                                quanhuyen = quanhuyen.replace('thị xã ', 'x ')
                                quanhuyen = quanhuyen.replace('quận ', 'q ')
                                quanhuyen = quanhuyen.replace('huyện ', 'h ')
                                if quanhuyen in addr:
                                    print('qh:', quanhuyen)
                                    if quanhuyen != new_name_qh:
                                        log += f'{cou}, Vi phạm quận huyện do quận huyện trong mục quận huyện khác trong địa chỉ chi tiết\n'
                                        cou = cou + 1
                                        back_qh = addr[-2]
                            else:
                                if quanhuyen not in QUAN.values():
                                    quanhuyen = quanhuyen.replace('thành phố ', '')
                                    quanhuyen = quanhuyen.replace('thị xã ', '')
                                    quanhuyen = quanhuyen.replace('quận ', '')
                                    quanhuyen = quanhuyen.replace('huyện ', '')
                                else:
                                    quanhuyen = quanhuyen.replace('thành phố ', 't ')
                                    quanhuyen = quanhuyen.replace('thị xã ', 'x ')
                                    quanhuyen = quanhuyen.replace('quận ', 'q ')
                                    quanhuyen = quanhuyen.replace('huyện ', 'h ')
                                if quanhuyen in addr:
                                    new_name_qh = new_name_qh[2:]
                                    print('qh:', quanhuyen)
                                    if quanhuyen != new_name_qh:
                                        print('aaaa:', new_name_qh)
                                        count = count + 1
                                        log += f'{cou}, Vi phạm quận huyện do quận huyện trong mục quận huyện khác trong địa chỉ chi tiết\n'
                                        cou = cou + 1
                                        back_qh = addr[-2]
        else:
            tinh_addr = 0
            ct_addr = 0
            for tinh_thanh in TINH_THANH.keys():
                if tinh_thanh.lower() in addr:
                    tinh_addr = tinh_addr + 1
                    print('tinhthanh:', tinh_thanh)
            if tinh_addr >= 1:
                print('có tỉnh')
                for cit in new_name_cit:
                    for ct in TINH_THANH[cit]:
                        print('ct:', ct)
                        if ct.lower() in addr:
                            ct_addr = ct_addr + 1
                            #add = new_addr.split(',')
                            #new_addr = new_addr.replace(add[-1], '')
                if ct_addr >= 1:
                    print('tỉnh trùng khớp')
                    if 'quận' in new_addr or 'q ' in new_addr or 'q.' in new_addr or 'thành phố' in new_addr or 'tp' in new_addr or 'thị xã' in new_addr or 'tx' in new_addr or 'huyện' in new_addr:
                        print(new_name_qh)
                        if new_name_qh in addr or new_name_qh[2:] in addr:
                            pass
                        else:
                            print(new_name_qh)
                            print('addr:', addr)
                            count = count + 1
                            log += f'{cou}, Không có quận huyện mục quận huyện trong địa chỉ chi tiết\n'
                            cou = cou + 1
                            back_qh = addr[-2]
                    else:
                        print(new_name_qh)
                        if new_name_qh in addr or new_name_qh[2:] in addr:
                            pass
                        else:
                            print(new_name_qh)
                            print('addr:', addr)
                            count = count + 1
                            log += f'{cou}, Không có quận huyện mục quận huyện trong địa chỉ chi tiết\n'
                            cou = cou + 1
                            back_qh = addr[-2]
            else:
                tinh_addr = 0
                for tinh_thanh in TINH_THANH.keys():
                    if tinh_thanh.lower() in name_cit.lower():
                       tinh_addr = tinh_addr + 1
                       print('tinhthanh:', tinh_thanh)
                if tinh_addr >= 1:
                   print('có tỉnh')
                else:
                   log += f'{cou}, Không có tỉnh thành\n\n'
                   cou = cou + 1
                # nếu trong addr không có tỉnh thì xét các huyện trong tỉnh có trong addr không
                for cit in new_name_cit:
                    for quanhuyen, tinhthanh in QUAN_HUYEN_TINH_THANH.items():
                        quanhuyen = quanhuyen.lower()
                        if tinhthanh == cit:
                            if 'quận' in new_addr or 'q ' in new_addr or 'q.' in new_addr or 'thành phố' in new_addr or 'tp' in new_addr or 'thị xã' in new_addr or 'tx' in new_addr or 'huyện' in new_addr:
                                quanhuyen = quanhuyen.replace('thành phố ', 't ')
                                quanhuyen = quanhuyen.replace('thị xã ', 'x ')
                                quanhuyen = quanhuyen.replace('quận ', 'q ')
                                quanhuyen = quanhuyen.replace('huyện ', 'h ')
                                if quanhuyen in addr:
                                    print('qh:', quanhuyen)
                                    if quanhuyen != new_name_qh:
                                        print('aaaa:', new_name_qh)
                                        count = count + 1
                                        print('vi phạm quận huyện')
                                        back_qh = addr[-2]
                            else:
                                if quanhuyen not in QUAN.values():
                                    quanhuyen = quanhuyen.replace('thành phố ', '')
                                    quanhuyen = quanhuyen.replace('thị xã ', '')
                                    quanhuyen = quanhuyen.replace('quận ', '')
                                    quanhuyen = quanhuyen.replace('huyện ', '')
                                else:
                                    quanhuyen = quanhuyen.replace('thành phố ', 't ')
                                    quanhuyen = quanhuyen.replace('thị xã ', 'x ')
                                    quanhuyen = quanhuyen.replace('quận ', 'q ')
                                    quanhuyen = quanhuyen.replace('huyện ', 'h ')
                                if quanhuyen in addr:
                                    new_name_qh = new_name_qh[2:]
                                    if quanhuyen != new_name_qh:
                                        print('aaaa:', new_name_qh)
                                        count = count + 1
                                        print('vi phạm quận huyện')
                                        back_qh = addr[-2]
    return count, back_qh, log
# index
def check_dk_6_1(dict):
    log = ""
    new_addr = dict['usc_address']
    new_addr = new_addr.lower()
    print(new_addr)
    if 'tp' in new_addr:
        print(11111)
    new_name_cit = dict['new_name_cit']
    new_name_qh = dict['new_name_qh']
    new_city = dict['new_city']
    new_name_cit = new_name_cit.split(',')
    new_name_qh = new_name_qh.lower()
    count = 0
    if ',' in new_name_qh:
        count = count + 1
    else:
        new_name_qh = new_name_qh.replace('thành phố ', 't ')
        new_name_qh = new_name_qh.replace('thị xã ', 'x ')
        new_name_qh = new_name_qh.replace('quận ', 'q ')
        new_name_qh = new_name_qh.replace('huyện ', 'h ')
    print('new_name_qh:', new_name_qh)
    addr = new_addr.lower()
    addr = addr.replace(', Việt Nam', '')
    addr = addr.replace('tỉnh ', '')
    addr = addr.replace('thành phố ', 't ')
    addr = addr.replace('thị xã ', 'x ')
    addr = addr.replace('quận ', 'q ')
    addr = addr.replace('huyện ', 'h ')
    addr = addr.replace('tx.', 'x ')
    addr = addr.replace('tp.', 't ')
    addr = addr.replace('tp', 't ')
    addr = addr.replace('tx', 'x ')
    addr = addr.replace('q.', 'q ')
    #addr = addr.replace('phường ', '')
    addr = addr.replace('.', ',')
    addr = addr.replace(', ', ',')
    addr = addr.replace('  ', ' ')
    addr = addr.replace('toà', 'tòa')
    addr = addr.replace('thủy', 'thuỷ')
    for quan_new in QUAN_NEW.values():
        quan_new = quan_new.lower()
        if quan_new in addr:
            addr = addr.replace(quan_new, list(QUAN_NEW.keys())[list(QUAN_NEW.values()).index(quan_new)])
            new_addr = new_addr.replace(quan_new, list(QUAN_NEW.keys())[list(QUAN_NEW.values()).index(quan_new)])
    print('addr:', addr)
    if new_name_cit[0] == '':
        if new_city != '0':
            count = count + 1
            print('không có tỉnh thành')
        else:
            count = count + 1
            print('tỉnh thành là toàn quốc')
    else:
        if new_name_qh == '':
            count = count + 1
            print('có tỉnh thành nhưng không có quận huyện')
        else:
            addr = addr.replace('–', ',')
            addr = addr.replace('-', ',')
            addr = addr.replace(' , ', ',')
            addr = addr.replace(' ,', ',')
            addr = addr.replace(', ', ',')
            if ',' in addr:
                addr = addr.split(',')
                # if 'tp ' in addr or 'tx ' in addr or 'q ' in addr or :
                tinh_addr = 0
                ct_addr = 0
                for tinh_thanh in TINH_THANH.keys():
                    if tinh_thanh.lower() == addr[-1]:
                        tinh_addr = tinh_addr + 1
                        print('tinhthanh:', tinh_thanh)
                if tinh_addr >= 1:
                    print('có tỉnh')
                    for cit in new_name_cit:
                        for ct in TINH_THANH[cit]:
                            print('ct:', ct)
                            if ct.lower() == addr[-1]:
                                ct_addr = ct_addr + 1
                                add = new_addr.split(',')
                                new_addr = new_addr.replace(add[-1], '')
                    if ct_addr >= 1:
                        print('tỉnh trùng khớp')
                        if 'quận' in new_addr or 'q ' in new_addr or 'q.' in new_addr or 'thành phố' in new_addr or 'tp' in new_addr or 'thị xã' in new_addr or 'tx' in new_addr or 'huyện' in new_addr:
                            if new_name_qh not in addr:
                                print('qh:', new_name_qh)
                                print('addr:', addr)
                                count = count + 1
                                print('có q trong add')
                        else:
                            new_name_qh = new_name_qh[2:]
                            print(new_name_qh)
                            if new_name_qh not in addr:
                                print(new_name_qh)
                                print('addr:', addr)
                                count = count + 1
                                print('không có q trong add')
                    else:
                        print('tỉnh không khớp')
                        # vi phạm
                        count = count + 1
                else:
                    print('không có tỉnh')
                    # nếu trong addr không có tỉnh thì xét các huyện trong tỉnh có trong addr không
                    for cit in new_name_cit:
                        for quanhuyen, tinhthanh in QUAN_HUYEN_TINH_THANH.items():
                            quanhuyen = quanhuyen.lower()
                            if tinhthanh == cit:
                                if 'quận' in new_addr or 'q ' in new_addr or 'q.' in new_addr or 'thành phố' in new_addr or 'tp' in new_addr or 'thị xã' in new_addr or 'tx' in new_addr or 'huyện' in new_addr:
                                    quanhuyen = quanhuyen.replace('thành phố ', 't ')
                                    quanhuyen = quanhuyen.replace('thị xã ', 'x ')
                                    quanhuyen = quanhuyen.replace('quận ', 'q ')
                                    quanhuyen = quanhuyen.replace('huyện ', 'h ')
                                    if quanhuyen in addr:
                                        print('qh:', quanhuyen)
                                        if quanhuyen != new_name_qh:
                                            print('aaaa:', new_name_qh)
                                            count = count + 1
                                            print('vi phạm quận huyện')
                                else:
                                    if quanhuyen not in QUAN.values():
                                        quanhuyen = quanhuyen.replace('thành phố ', '')
                                        quanhuyen = quanhuyen.replace('thị xã ', '')
                                        quanhuyen = quanhuyen.replace('quận ', '')
                                        quanhuyen = quanhuyen.replace('huyện ', '')
                                    else:
                                        quanhuyen = quanhuyen.replace('thành phố ', 't ')
                                        quanhuyen = quanhuyen.replace('thị xã ', 'x ')
                                        quanhuyen = quanhuyen.replace('quận ', 'q ')
                                        quanhuyen = quanhuyen.replace('huyện ', 'h ')
                                    if quanhuyen in addr:
                                        new_name_qh = new_name_qh[2:]
                                        if quanhuyen != new_name_qh:
                                            count = count + 1
                                            print('vi phạm quận huyện')
            else:
                tinh_addr = 0
                ct_addr = 0
                for tinh_thanh in TINH_THANH.keys():
                    if tinh_thanh.lower() in addr:
                        tinh_addr = tinh_addr + 1
                        print('tinhthanh:', tinh_thanh)
                if tinh_addr >= 1:
                    print('có tỉnh')
                    for cit in new_name_cit:
                        for ct in TINH_THANH[cit]:
                            print('ct:', ct)
                            if ct.lower() in addr:
                                ct_addr = ct_addr + 1
                                #add = new_addr.split(',')
                                #new_addr = new_addr.replace(add[-1], '')
                    if ct_addr >= 1:
                        print('tỉnh trùng khớp')
                        if 'quận' in new_addr or 'q ' in new_addr or 'q.' in new_addr or 'thành phố' in new_addr or 'tp' in new_addr or 'thị xã' in new_addr or 'tx' in new_addr or 'huyện' in new_addr:
                            if new_name_qh not in addr:
                                count = count + 1
                                print('có q trong add')
                        else:
                            new_name_qh = new_name_qh[2:]
                            print(new_name_qh)
                            if new_name_qh not in addr:
                                print(new_name_qh)
                                print('addr:', addr)
                                count = count + 1
                                print('không có q trong add')
                else:
                    print('không có tỉnh')
                    # nếu trong addr không có tỉnh thì xét các huyện trong tỉnh có trong addr không
                    for cit in new_name_cit:
                        for quanhuyen, tinhthanh in QUAN_HUYEN_TINH_THANH.items():
                            quanhuyen = quanhuyen.lower()
                            if tinhthanh == cit:
                                if 'quận' in new_addr or 'q ' in new_addr or 'q.' in new_addr or 'thành phố' in new_addr or 'tp' in new_addr or 'thị xã' in new_addr or 'tx' in new_addr or 'huyện' in new_addr:
                                    quanhuyen = quanhuyen.replace('thành phố ', 't ')
                                    quanhuyen = quanhuyen.replace('thị xã ', 'x ')
                                    quanhuyen = quanhuyen.replace('quận ', 'q ')
                                    quanhuyen = quanhuyen.replace('huyện ', 'h ')
                                    if quanhuyen in addr:
                                        print('qh:', quanhuyen)
                                        if quanhuyen != new_name_qh:
                                            print('aaaa:', new_name_qh)
                                            count = count + 1
                                            print('vi phạm quận huyện')
                                else:
                                    if quanhuyen not in QUAN.values():
                                        quanhuyen = quanhuyen.replace('thành phố ', '')
                                        quanhuyen = quanhuyen.replace('thị xã ', '')
                                        quanhuyen = quanhuyen.replace('quận ', '')
                                        quanhuyen = quanhuyen.replace('huyện ', '')
                                    else:
                                        quanhuyen = quanhuyen.replace('quận ', 'q ')
                                    if quanhuyen in addr:
                                        new_name_qh = new_name_qh[2:]
                                        print('qh:', quanhuyen)
                                        if quanhuyen != new_name_qh:
                                            count = count + 1
                                            print('vi phạm quận huyện')

    return count




# kiểm tra điều kiện 7
# tên NTD không được phép có các từ tuyển dụng, tìm việc, việc làm
def check_dk_7(dict):
    log = ""
    ten_congty = dict['usc_company']
    count = 0
    keys = ['tuyển dụng', 'tìm việc làm', 'việc làm']
    ten_congty = ten_congty.lower()
    for key in keys:
        if key in ten_congty:
            count = count + 1
            ten_congty.replace(key, "")
    if ten_congty == "":
        count = count + 1
        log = "Không rõ công ty\n"
    return count, ten_congty, log

# kiểm tra điều kiện 8
# tên công ty phải rõ ràng
def check_dk_8(dict):
    log =  ""
    ten_congty = dict['usc_company']
    count = 1
    ten_congty = ten_congty.lower()
    keys = ['công ty cổ phần', 'công ty cp', 'công ty', 'tập đoàn', 'trung tâm', 'hệ thống', 'cty', 'group', 'tnhh', 'bất động sản', 'shinhan fc', 'aliexpress'] 
    for key in keys:
        if key in ten_congty:
            count = count - 1
            ten_congty.replace(key, "")
            continue
    if count < 0 and ten_congty != "":
        count = 0
    else:
        log = "Tên công ty không rõ ràng\n"
        count = 1
    return count, log

# kiểm tra điều kiện 9
# Tiêu đề không chứa các ký tự đặc biệt, không chứa lương, sdt
def check_dk_9(dict):
    log = ""
    cou = 1
    new_title = dict['new_title']
    new_addr = dict['usc_address']
    new_name_cit = dict['new_name_cit']
    new_name_cit = new_name_cit.split(',')
    new_name_qh = dict['new_name_qh']
    new_name_qh = new_name_qh.split(',')
    count = 0
    back_title = new_title
    # kiểm tra sdt
    patterns = ['[0-9]{10}', '[0-9]{5} [0-9]{5}',
                '[0-9]{4} [0-9]{3} [0-9]{3}',
                '[0-9]{4}-[0-9]{3}-[0-9]{3}',
                '[0-9]{3}-[0-9]{3}-[0-9]{4}']
    for pattern in patterns:
        if (re.search(pattern, back_title)):
            count = count + 1
            log += f'{cou}, Tiêu đề chứa số điện thoại\n'
            cou = cou + 1
            re.sub(pattern, '', back_title, flags=re.IGNORECASE)

    # kiểm tra ký tự đặc biệt
    #pattern = r'[^A-Za-z0-9\s]{2,}'
    pattern = r'[^\\/\w\s,.;<>]{1,}'
    #pattern = r'[^0-9A-Za-z\s]'
    if (re.search(r'[^\\/\w\s,.;<>]{1,}', back_title)):
        count = count + 1
        log +=  f'{cou}, Tiêu đề chứa ký tự đặc biệt\n. --Đã sửa lại--\n'
        cou = cou + 1
        back_title = re.sub(pattern, '', back_title)
    # kiểm tra địa chỉ
    if 'chi nhánh' in new_title.lower():
        count = count + 1
        log += f'{cou}, Chi nhánh trong tiêu đề\n'
        cou = cou + 1
        back_title = re.sub(r"chi nhánh", "", back_title)
#    if 'toàn quốc' in new_title.lower():
#        count = count + 1
#        log += f'{cou}, Địa điểm là không được là toàn quốc, phải là địa điểm cụ thể'
#        cou = cou + 1
#        back_title = re.sub(r"toàn quốc", "", back_title)
    title = new_title.lower()
    title = underthesea.word_tokenize(title, fixed_words=FIXED_WORDS, format="text")
    print('titt:', title)
    title = title.split()
    print('tit:', title)
    for tinhthanh in TINH_FIXED:
        tinhthanh = tinhthanh.replace("-", " ")
        if tinhthanh in back_title:
            count = count + 1
            log += f'{cou},Tỉnh trong tiêu đề\n'
            cou = cou + 1
            pattern = re.compile(re.escape(tinhthanh), re.IGNORECASE)
            back_title = pattern.sub("", back_title)
    for quanhuyen in QUAN_FIXED:
        quanhuyen = quanhuyen.replace("-", " ")
        if quanhuyen in back_title:
            count = count + 1
            print(quanhuyen)
            log += f'{cou}, Huyện trong tiêu đề\n'
            cou = cou + 1
            pattern = re.compile(re.escape(quanhuyen), re.IGNORECASE)
            back_title = pattern.sub("", back_title)
    # kiểm tra lương
    pats = TITLE_JOBPOSTING
    for pat in pats:
        regex = re.compile(pat, flags=re.IGNORECASE)
        if regex.search(back_title.lower()):
        # 3. Thay mọi match bằng một khoảng trắng
           back_title = regex.sub(" ", back_title)
        # 3. Thay mọi match bằng một khoảng trắng
           back_title = regex.sub(" ", back_title)
    # (Optional) Xoá dư khoảng trắng thừa thành một khoảng giữa các từ
           back_title = re.sub(r'\s+', ' ', back_title).strip()
           count = count + 1
           log += f'{cou}, Lương trong tiêu đề\n'
           cou = cou + 1
    return count, back_title, log


def check_special(texts):
    count = 0
    text_special = ['!', '@', '#', '$', '%', '^',
                    '&', '*', '(', ')',
                    '_', '+', '=', '[', ']',
                    '{', '}', '|', ':', ';',
                    '<', '>', '?', '/']
    for special in text_special:
        for text in texts:
            if special == text:
                count = count + 1
    return count


#dk 9 cho index

def check_dk_9_1(dict):
    new_title = dict['new_title']
    new_addr = dict['usc_address']
    new_name_cit = dict['new_name_cit']
    new_name_cit = new_name_cit.split(',')
    new_name_qh = dict['new_name_qh']
    new_name_qh = new_name_qh.split(',')
    count = 0
    # kiểm tra sdt
    patterns = ['[0-9]{10}', '[0-9]{5} [0-9]{5}',
                '[0-9]{4} [0-9]{3} [0-9]{3}',
                '[0-9]{4}-[0-9]{3}-[0-9]{3}',
                '[0-9]{3}-[0-9]{3}-[0-9]{4}']
    for pattern in patterns:
        if (re.search(pattern, new_title)):
            count = count + 1
            print('tiêu đề chứa số điện thoại')
    # kiểm tra ký tự đặc biệt
    if (re.search(r'[!@#$%^&*()?":;{}|<>_=/+\|]', new_title)):
        count = count + 1
        print('tiêu đề chứa ký tự đặc biệt')
    # kiểm tra địa chỉ
    if 'chi nhánh' in new_title.lower():
        count = count + 1
        print('chi nhánh trong tiêu đề')
    title = new_title.lower()
    title = underthesea.word_tokenize(title, fixed_words=FIXED_WORDS, format="text")
    title = title.split()
    print('tit:', title)
    for tinhthanh in TINH_FIXED:
        if tinhthanh in title:
            count = count + 1
            print('tỉnh trong tiêu đề')
    for quanhuyen in QUAN_FIXED:
        if quanhuyen in title:
            count = count + 1
            print(quanhuyen)
            print('huyện trong tiêu đề')
    # kiểm tra lương
    pats = TITLE_INDEX
    new_title = new_title.lower()
    print('new_title:', new_title)
    for pat in pats:
        if (re.findall(pat, new_title)) :
            if 'tr' in re.findall(pat, new_title)[0] or 'm' in re.findall(pat, new_title)[0]:
                if re.findall(pat, new_title)[0][-2:] == 'tr' or re.findall(pat, new_title)[0][-1] == 'm' or 'triệu' in re.findall(pat, new_title)[0]:
                    count = count + 1
                    print('lương trong tiêu đề')
            else:
                count = count + 1
                print('lương trong tiêu đề')
    return count

text_special = ['!', '#', '$', '%', '^','*','?'
                    '&', '{', '}', '|',
                    '_', '+', '=', '@', '[', ']',
                    ':', ';',
                    '<', '>', '/', '(', ')',]
def check_special(texts):
    count = 0
    for special in text_special:
        for text in texts:
            if special == text:
                count = count + 1
    return count

def clear_special(texts):
    for special in text_special:
        if special in texts:
            texts.replace(special, '')
# kiểm tra điều kiện 10
# Nội dung tin không chứa các ký tự đặc biệt quá 5%
def check_dk_10(dict):
    log =  ""
    status = 0
    new_mota = dict['new_mota']
    new_yeucau = dict['new_yeucau']
    new_quyenloi = dict['new_quyenloi']
    new_mota = clean_text(new_mota)
    new_quyenloi = clean_text(new_quyenloi)
    new_yeucau = clean_text(new_yeucau)
    special_mota = check_special(new_mota)
    special_yeucau = check_special(new_yeucau)
    special_quyenloi = check_special(new_quyenloi)
    if (len(new_mota)+len(new_yeucau)+len(new_quyenloi)) != 0:
        ratio = (special_mota+special_yeucau+special_quyenloi)/(len(new_mota)+len(new_yeucau)+len(new_quyenloi))
    else:
        ratio = 0.06
        log += 'Khong co 1 trong các mục mota quyenloi yeucau\n'
        status = 2

    print('mota:', new_mota)
    print('yeucau:', new_yeucau)
    print('quyenloi:', new_quyenloi)
    print('tile:', ratio)
    if ratio < 0.05:
        return 0, new_mota, new_quyenloi, new_yeucau, log
    else:
        status = 1
        for special in text_special:
           if special in new_mota:
              new_mota  = new_mota.replace(special, '')
           if special in new_quyenloi:
              new_quyenloi = new_quyenloi.replace(special, '')
           if special in new_yeucau:
              new_yeucau = new_yeucau.replace(special, '')
           special_mota = check_special(new_mota)
           special_yeucau = check_special(new_yeucau)
           special_quyenloi = check_special(new_quyenloi)
           if (len(new_mota)+len(new_yeucau)+len(new_quyenloi)) != 0:
              ratio = (special_mota+special_yeucau+special_quyenloi)/(len(new_mota)+len(new_yeucau)+len(new_quyenloi))
              if(ratio < 0.05):
                  break
           else:
               ratio = 0.06
               log +=  'Khong co 1 trong các mục mota quyenloi yeucau\n'
               status = 2
               break
    return status, new_mota, new_quyenloi, new_yeucau, log


def load_bert():
    v_phobert = AutoModel.from_pretrained("vinai/phobert-base")
    v_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    return v_phobert, v_tokenizer

# Hàm chuẩn hoá câu
def standardize_data(row):
    # Xóa dấu chấm, phẩy, hỏi ở cuối câu
    row = re.sub(r"[\.,\?]+$-", "", row)
    # Xóa tất cả dấu chấm, phẩy, chấm phẩy, chấm thang, ... trong câu
    row = row.replace(",", " ").replace(".", " ") \
        .replace(";", " ").replace("“", " ") \
        .replace(":", " ").replace("”", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("?", " ") \
        .replace("-", " ").replace("?", " ")
    row = row.strip().lower()
    return row

# Hàm load danh sách các từ vô nghĩa: lắm, ạ, à, bị, vì..
def load_stopwords():
    sw = []
    with open("vietnamese-stopwords.txt", encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        sw.append(line.replace("\n",""))
    return sw

# Hàm tạo ra bert features
def make_bert_features(text, max_len):
    #global phobert, sw
    phobert, tokenizer = load_bert()
    sw = load_stopwords()
    text = standardize_data(text)
    text = text.split()
    print("text:", text)
    filtered_sentence = [w for w in text if w not in sw]
    print("filtered_sentence:", filtered_sentence)
    text = " ".join(filtered_sentence)
    text = underthesea.word_tokenize(text, format="text")
    text = [w for w in text if w not in sw]
    text = "".join(text)
    print("text:", text)
    text = tokenizer.encode(text)
    padded = numpy.array([text[:max_len] if len(text) >= max_len else (text + [1] * (max_len - len(text)))])
    attention_mask = numpy.where(padded == 1, 0, 1)
    padded = torch.tensor(padded).to(torch.long)
    attention_mask = torch.tensor(attention_mask)
    with torch.no_grad():
        last_hidden_states = phobert(input_ids= padded, attention_mask=attention_mask)
    features = last_hidden_states[0][:, 0, :].numpy()
    print(features.shape)
    return features

def standardize_sentence(title, description):
    sw = load_stopwords()
    title = standardize_data(title)
    title = title.split()
    description = standardize_data(description)
    description = description.split()
    print("title:", title)
    print("description:", description)
    filtered_title = [w for w in title if w not in sw]
    print("filtered_title:", filtered_title)
    title = " ".join(filtered_title)
    filtered_description = [w for w in description if w not in sw]
    print("filtered_description:", filtered_description)
    title = " ".join(filtered_title)
    description = " ".join(filtered_description)
    print("title:", title)
    print("description", description)

    return title, description

# kiểm tra điều kiện 11
# Nội dung tin phải có nghĩa, k được sơ sài, nội dung tiêu đề phải khớp với nôi dung mô tả.
def  tfidf(title, description):
    vectorizer = TfidfVectorizer(
    lowercase=True,
    token_pattern=r"(?u)\b\w\w+\b",   # chỉ lấy token >=2 ký tự
    stop_words=["và","của","có","là","cho","đến"]  # ví dụ stop‑words
)
    title = [title]
    description = [description]
# 2) Fit lên toàn bộ văn bản (cả titles + descriptions)
    all_texts = title + description
    vectorizer.fit(all_texts)
# 3) Chuyển riêng từng phần
    tfidf_titles       = vectorizer.transform(title)        # shape: (n_titles, n_features)
    tfidf_descriptions = vectorizer.transform(description)  # shape: (n_descs, n_features)
    sim = cosine_similarity(
        tfidf_titles[0],
        tfidf_descriptions[0]
    )[0][0]
    
    return sim

def hybrid_similarity(title, description, alpha = 0.5):
    sw = load_stopwords()
    model = SentenceTransformer('all-mpnet-base-v2')
        # --- Chuẩn hóa & tokenization ---
    # Use the standardize_sentence function to get filtered tokens directly
    title_tokens, desc_tokens = standardize_sentence(title, description)

    # --- BM25 part ---
    bm25 = BM25Okapi([desc_tokens])
    bm25_score = bm25.get_scores(title_tokens)[0] # Corrected method to get_scores

    # --- Embedding part ---
    emb_title = model.encode(
        title_tokens, convert_to_tensor=True
    )
    emb_desc  = model.encode(
        desc_tokens, convert_to_tensor=True
    )
    cosine_sim = util.cos_sim(emb_title, emb_desc).item()

    # --- Hybrid score ---
    hybrid = alpha * bm25_score + (1 - alpha) * cosine_sim

    return {
        "bm25_score": bm25_score,
        "cosine_similarity": cosine_sim,
        "hybrid_score": hybrid
    }

def final_score(title, description, alpha = 0.75):
    scores = hybrid_similarity(title, description)
    tf = scores["cosine_similarity"]
    sim = tfidf(title, description)
    final_scores = tf*alpha + (1-alpha)*sim

    return final_scores >= 0.3

def check_dk_11(dict):
    log = ""
    tieude = dict['new_title']
    mota = dict['new_mota']
    mota = clean_text(mota)
    tieude = clean_text(tieude)
    print('tieude11:', tieude)
    print('mota11:', mota)
    check_dk = final_score(tieude, mota)
    if check_dk:
        return 0, log
    else:
        log += "Không được tiêu đề một kiểu, mô tả một kiểu"
        return 1, log

def check_dk_12(dict):
    tieude = dict['new_title']
    mota = dict['new_mota']
    count = 0
    mota = clean_text(mota)
    tieude = clean_text(tieude)
    tieude = tieude.lower()
    mota = mota.lower()
    tieude = re.sub(r'[!@#$%^&*()–?,.":;{}|<>_=/+\|]', '', tieude)
    mota = re.sub(r'[!@#$%^&*().,–?":;{}|<>_=/+\|]', '', mota)
    #mota = re.sub(r'\s+', ' ', mota)
    fixed_words = ['chăm sóc khách hàng', 'làm thêm', 'thương mại điện tử', 'bất động sản', 'bán hàng', 'trợ giảng', 'thực tập sinh', 'e commerce', 'live stream']
    tieude = underthesea.word_tokenize(tieude, fixed_words=fixed_words, format="text")
    mota = underthesea.word_tokenize(mota, fixed_words=fixed_words, format="text")
    tieude = tieude.split()
    mota = mota.split()
    for tit in tieude:
        if tit in MAP.keys():
            print('tit:', tit)
            for value in MAP[tit]:
                if value in mota:
                    print('value:', value)
                    count = count + 1
        else:
            if tit in mota:
                print('tit not in MAP.key() but title in mota')
                count = count + 1
    print('tieude:', tieude)
    print('mota:', mota)
    if count > 0:
        return 0
    return 1

def is_too_short(text: str, min_words: int = 30, min_chars: int = 200) -> bool:
    return len(text.split()) < min_words or len(text) < min_chars

def vietnamese_readability(text: str) -> dict:
    # 1. Tách câu và từ
    sentences = sent_tokenize(text)
    words_per_sent = [word_tokenize(s) for s in sentences]

    num_sents = len(sentences)
    num_words = sum(len(ws) for ws in words_per_sent)
    if num_sents == 0 or num_words == 0:
        return {
            "avg_words_per_sentence": 0,
            "avg_chars_per_word": 0,
            "polysyllable_ratio": 0,
            "readability_score": 0
        }

    # 2. Tính trung bình số từ / câu
    avg_wps = num_words / num_sents

    # 3. Tính trung bình số ký tự / từ
    total_chars = sum(len(w) for ws in words_per_sent for w in ws)
    avg_cpw = total_chars / num_words

    # 4. Tỉ lệ từ dài (giả sử >=5 ký tự xem là “đa âm”)
    poly_count = sum(1 for ws in words_per_sent for w in ws if len(w) >= 5)
    polysyllable_ratio = poly_count / num_words

    # 5. Công thức điểm readability (ví dụ tham khảo)
    #    Càng nhiều words/sentence, chars/word, polysyllables → càng khó đọc → trừ vào 100
    score = 100 \
            - (avg_wps * 1.5) \
            - (avg_cpw * 3) \
            - (polysyllable_ratio * 20)
    # giới hạn trong 0–100
    score = max(0, min(100, score))

    return {
        "avg_words_per_sentence": round(avg_wps, 2),
        "avg_chars_per_word": round(avg_cpw, 2),
        "polysyllable_ratio": round(polysyllable_ratio, 2),
        "readability_score": round(score, 2)
    }


def is_poor_vn_readability(text: str,
                           low_thresh: float = 30,
                           high_thresh: float = 80) -> bool:
    """
    Trả về True nếu readability_score < low_thresh (quá khó)
    hoặc > high_thresh (quá đơn giản / có thể vô nghĩa).
    """
    metrics = vietnamese_readability(text)
    sc = metrics["readability_score"]
    return sc < low_thresh or sc > high_thresh

#def grammar_error_rate(text: str) -> float:
    tool = language_tool_python.LanguageTool('vi')  # hoặc 'en' nếu tiếng Anh
    matches = tool.check(text)
    return len(matches) / max(len(text.split()), 1)

#def is_high_error_rate(text: str, max_rate: float = 0.05) -> bool:
    return grammar_error_rate(text) > max_rate

from underthesea import word_tokenize, pos_tag

def grammar_error_rate(text: str) -> float:
    tokens = word_tokenize(text)
    tags   = pos_tag(text)
    # Ví dụ: tính tỉ lệ token được gán POS 'X' (unknown) hay punctuation sai chỗ
    num_unknown = sum(1 for _, t in tags if t == 'X')
    return num_unknown / max(len(tokens), 1)

def is_high_error_rate(text: str, max_rate: float = 0.05) -> bool:
    return grammar_error_rate(text) > max_rate


def perplexity(text: str) -> float:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model     = AutoModelForCausalLM.from_pretrained("gpt2")
    enc = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        loss = model(**enc, labels=enc["input_ids"]).loss
    return torch.exp(loss).item()

def is_gibberish(text: str, max_ppl: float = 1000) -> bool:
    return perplexity(text) > max_ppl

def type_token_ratio(text: str) -> float:
    tokens = text.lower().split()
    return len(set(tokens)) / len(tokens)

def is_low_diversity(text: str, min_ttr: float = 0.3) -> bool:
    return type_token_ratio(text) < min_ttr

def mean_coherence(text: str) -> float:
    model = SentenceTransformer('all-mpnet-base-v2')
    sentences = [s for s in text.split('.') if s.strip()]
    embs = model.encode(sentences, convert_to_tensor=True)
    sims = []
    for i in range(len(embs)-1):
        sims.append(util.cos_sim(embs[i], embs[i+1]).item())
    return sum(sims) / max(len(sims), 1)

def is_low_coherence(text: str, min_coh: float = 0.2) -> bool:
    return mean_coherence(text) < min_coh

def is_poor_description(text: str) -> bool:
    checks = [
        is_too_short(text),
        is_poor_vn_readability(text),
        is_high_error_rate(text),
        is_gibberish(text),
        is_low_diversity(text),
        is_low_coherence(text), 
    ]
    # Nếu >= 2/3 tiêu chí báo xấu, xem là "kém chất lượng"
    return sum(checks) >= 4 or checks[0]
# kiểm tra điều kiện sơ sài
def check_dk_bosung(dict):
    yeucau = dict['new_yeucau']
    mota = dict['new_mota']
    quyenloi = dict['new_quyenloi']
    count = 0
    ql = 0
    mota = clean_text(mota)
    yeucau = clean_text(yeucau)
    quyenloi = clean_text(quyenloi)
    yeucau = yeucau.lower()
    mota = mota.lower()
    quyenloi = quyenloi.lower()
    stop_word = ['và', 'của', 'có', 'các']
    for w in stop_word:
        mota = mota.replace(w, ' ')
        quyenloi = quyenloi.replace(w, ' ')
        yeucau = yeucau.replace(w, ' ')
    yeucau = re.sub(r'[!@#$%^&*()–?,.":;{}|<>_=/+\|-]', '', yeucau)
    mota = re.sub(r'[!@#$%^&*().,–?":;{}|<>_=/+\|-]', '', mota)
    quyenloi = re.sub(r'[!@#$%^&*().,–?":;{}|<>_=/+\|-]', '', quyenloi)
    return is_poor_description(yeucau), is_poor_description(mota), is_poor_description(quyenloi)
    

class ErrorModel:
    def __init__(self, code, message):
        self.code = code
        self.message = message

class ResponseModel:
    # Trả về phản hồi gồm data và error
    def __init__(self, data, error):
        self.data = data
        self.error = error

class DataModel:
    def __init__(self, new_id, result, message, dictionary, status):
        self.new_id = new_id
        self.num_error = result
        self.message = message
        self.dictionary = dictionary
        self.status = status
