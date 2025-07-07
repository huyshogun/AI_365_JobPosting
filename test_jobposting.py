from index_jobsposting.utils import*
from index_jobsposting.config import QUAN_HUYEN
from crawl_data.use_llm import *
import requests
import json
from flask import Flask, request, jsonify, abort
import time
from IPython.display import Markdown
from dotenv import load_dotenv
import os
load_dotenv()
#id = 1
app = Flask(__name__)
VALID_API_KEY = os.getenv("API_KEY")
def check_api_key():
    # Đọc API key client gửi lên: ở đây dùng header "X-API-KEY"
    client_key = request.headers.get("X-API-KEY")
    if not client_key or client_key != VALID_API_KEY:
        # Nếu thiếu hoặc sai key → trả 401 Unauthorized
        abort(401, description="Invalid or missing API key")
 
@app.route('/jobsposting_new', methods=['POST', 'GET'])
def jobsposting_new():
    data_body = request.form
    print(data_body)
    new_id = data_body.get('new_id')
    message = {}
    status = 1
    if status == 1:
     try:
        api = f"https://timviec365.vn/api_app/chi_tiet_tin.php?newid={new_id}"
#        with open(f"test_post/test_post_{new_id}.json", "r", encoding="utf-8") as f:
#           data = json.load(f)
        data = requests.get(api).json()
        if data['data']['data'] != None :
            dictionary = data['data']['data']
#            mota = data['data']['data']['new_mota']
#            quyenloi = data['data']['data']['new_quyenloi']
#            yeucau = data['data']['data']['new_yeucau']
#            data['data']['data']['new_mota'] = re.sub(r'[^\\\w\s,.;]{2,}', '', mota)
#            data['data']['data']['new_quyenloi'] = re.sub(r'[^\\\w\s,.;]{2,}', '', quyenloi)
#            data['data']['data']['new_yeucau'] = re.sub(r'[^\\\w\s,.;]{2,}', '', yeucau)  
            print(dictionary)
            one = time.time()
            count_1, new_cit, log_1 = check_dk_1(dictionary)
            data['data']['data']['new_city'] = new_cit
                
            print('1:', count_1)
            count_2, back_addr, data['data']['data']['new_title'], data['data']['data']['new_mota'], log_2 = check_dk_2(dictionary)
            if count_2 == 0:
                data['data']['data']['usc_address'] = back_addr
            else:
                status = 0
            print('2:', count_2)
            count_3, log_3 = check_dk_3(dictionary)
            if count_3 != 0:
                status = 0
            print('3:', count_3)
            count_4, log_4 = check_dk_4(dictionary)
            if count_4 != 0:
                status = 0
            print('4:', count_4)
            count_5, log_5 = check_dk_5(dictionary)
            if count_5 != 0:
                status = 0
            print('5:', count_5)
            count_6, back_qh, log_6 = check_dk_6(dictionary)
            if count_6 != 0:
                data['data']['data']['new_name_qh'] = back_qh
            print('6:', count_6)
            count_7, ten_congty,  log_7 = check_dk_7(dictionary)
            if count_7 != 0:
                if ten_congty == "":
                    status = 0
            print('7:', count_7)
            count_8, log_8 = check_dk_8(dictionary)
            if count_8 != 0:
                status = 0
            print('8:', count_8)
 #           if status != 0: #Test thử
            if status == 0:
                log_9 = log_10 = log_11 = log_12 = ""
                result = count_1 + count_2 + count_3 + count_4 + count_5 + count_6 + count_7 + count_8
            else:
              count_9, back_title, log_9 = check_dk_9(data['data']['data'])
              if count_9 != 0:
                data['data']['data']['new_title'] = back_title
                status = 2
              print('9:', count_9)
              count_10, back_mota, back_quyenloi, back_yeucau, log_10 = check_dk_10(dictionary)
              if count_10 == 1:
                data['data']['data']['new_mota'] = back_mota
                data['data']['data']['new_quyenloi'] = back_quyenloi
                data['data']['data']['new_yeucau'] = back_yeucau
                status = 4

              print('10:', count_10)
              two = time.time()
              has_mota = False
              use_llm  = []
              count_11, log_11 = check_dk_11(data['data']['data'])
              if count_11 != 0:
                use_llm.append("Tiêu đề:\n" + data['data']['data']['new_title'])
                use_llm.append("Mô tả:\n" + data['data']['data']['new_mota'] + "\n")
                has_mota = True
                status = 4                


 
              print('11:', count_11)
              check_yeucau, check_mota, check_quyenloi = check_dk_bosung(data['data']['data'])
              log_12 = ""
              count_12 = 0
#             message_12 = "không vi phạm điều kiện 12"
              if check_mota:
               if not has_mota:
                  use_llm.append("Mô tả:\n" + data['data']['data']['new_mota'] + "\n")
                  status = 4
#               message_12 =  'vi phạm điều kiện 12'
               count_12 = 1
               log_12 +=  "Mô tả lôm côm, sơ sài. Đã viết lại mô tả\n"
              if check_yeucau:
               use_llm.append("Yêu cầu:\n" + data['data']['data']['new_yeucau'] + "\n")
               status = 4
#               message_12 =  'vi phạm điều kiện 12'
               count_12 = 1
               log_12 +=  "Yêu cầu lôm côm, sơ sài, cần phải viết lại. Đã viết lại yêu cầu\n"
               status = 4
              if check_quyenloi:
               use_llm.append("Quyền lợi:\n" + data['data']['data']['new_quyenloi']  +  "\n")
               status = 4
 #              message_12 =  'vi phạm điều kiện 12'
               count_12 = 1
               log_12 +=  "Quyền lợi lôm côm, sơ sài, cần phải viết lại. Đã viết lại quyền lợi\n"
               status = 4
              origin_text = ""
              if len(use_llm)  > 0:
               for text in use_llm:
                   origin_text += text
               fixed_dict = rewrite(origin_text)
               value = get_ci(fixed_dict, "Tiêu đề")
               if value != "":
                   data['data']['data']['new_title'] = "<p>" + value.replace("\n", "<br />") + "</p>"
               value = get_ci(fixed_dict, "Vị trí")
               if value != "":
                   data['data']['data']['new_title'] = "<p>" + value.replace("\n", "<br />") + "</p>"
               value = get_ci(fixed_dict, "Mô tả công việc")
               if value != "":
                   data['data']['data']['new_mota'] = "<p>" + value.replace("\n", "<br />") + "</p>"
               value = get_ci(fixed_dict, "Mô tả")
               if value != "":
                   data['data']['data']['new_mota'] = "<p>" + value.replace("\n", "<br />") + "</p>"
               value = get_ci(fixed_dict, "Yêu cầu ứng viên")
               if value != "":
                   data['data']['data']['new_yeucau'] = "<p>" + value.replace("\n", "<br />") + "</p>"
               value = get_ci(fixed_dict, "Yêu cầu")
               if value != "":
                   data['data']['data']['new_yeucau'] = "<p>" + value.replace("\n", "<br />") + "</p>"
               value = get_ci(fixed_dict, "Quyền lợi")
               if value != "":
                   data['data']['data']['new_quyenloi'] = "<p>" + value.replace("\n", "<br />") + "</p>"              
              if not check_mota:
                   mota = data['data']['data']['new_mota']
                   data['data']['data']['new_mota'] = re.sub(r'[^\\/\w\s,.;<>]{1,}', '', mota)
              if not check_quyenloi:
                   quyenloi = data['data']['data']['new_quyenloi']
                   data['data']['data']['new_quyenloi'] = re.sub(r'[^\\/\w\s,.;<>]{1,}', '', quyenloi)
              if not check_yeucau:
                   yeucau = data['data']['data']['new_yeucau']
                   data['data']['data']['new_yeucau'] = re.sub(r'[^\\/\w\s,.;<>]{1,}', '', yeucau)
              three = time.time()
              print(two - one)
              print(three - two)
              result = count_1 + count_2 + count_3 + count_4 + count_5 + count_6 + count_7 + count_8 + count_9 + count_10 + count_11 + count_12
        
            message['Điều kiện 1'] = log_1
            message['Điều kiện 2'] = log_2
            message['Điều kiện 3'] = log_3
            message['Điều kiện 4'] = log_4
            message['Điều kiện 5'] = log_5
            message['Điều kiện 6'] = log_6
            message['Điều kiện 7'] = log_7
            message['Điều kiện 8'] = log_8
            message['Điều kiện 9'] = log_9
            message['Điều kiện 10'] = log_10
            message['Điều kiện 11'] = log_11
            message['Điều kiện 12'] = log_12
            
            print(count_1)
            print(count_2)
            print(count_3)
            print(count_4)
            print(count_5)
            print(count_6)
            print(count_7)
            print(count_8)

        else:
            result = -1
            message['message'] = 'công ty không tồn tại'
            status = 0
        if  status == 0:
            new_status = "Trả lại cho NTD để sửa lại các lỗi ở phần message.\n"
        elif status == 1:
            new_status = "Không gặp lỗi gì.\n"
        elif status == 2:
            new_status = "Đã sửa lỗi tiêu đề ở điều kiện 9.\n"
        elif status == 4:
            new_status = "Đã viết lại Tiêu đê hoặc Mô tả hoặc Quyền lợi hoặc Yêu cầu.\n"
        data = DataModel(new_id, result, message, data['data']['data'], new_status)
        error = None
     except Exception as err:
        print(err)
        error = ErrorModel(200, message)
        data = None
        
    if data is not None:
        data = vars(data)
    if error is not None:
        error = vars(error)
    response = ResponseModel(data, error)
    with open(f"fix_post/fix_{new_id}.json", "w", encoding="utf-8") as f:
        json.dump(vars(response), f, ensure_ascii=False, indent=4)
    return jsonify(vars(response)), 200
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8008)


