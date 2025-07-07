import re
import openai
from dotenv import load_dotenv
import os

load_dotenv()

# 1. Cấu hình OpenRouter với API key trực tiếp
# API_KEY = os.getenv("API_KEY_OPENROUTER")
API_KEY = os.getenv("API_LLM_KEY")
# Instantiate the client using the new syntax
client = openai.OpenAI(
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1"
)
# 2. Chọn model meta-llama/llama-4-maverick:free
MODEL1 = "meta-llama/llama-4-maverick:free"
MODEL2  = "deepseek/deepseek-r1:free"
# 3. Định nghĩa hàm hỏi đáp chatbot
def generate_answer(client: openai.OpenAI, question: str) -> str:
    """
    Gửi câu hỏi tới model và trả về câu trả lời.
    Requires an initialized openai.OpenAI client object.
    """
    response = client.chat.completions.create( # Use the client instance
        model=MODEL2,
        messages=[
            {"role": "system", "content": (
                "Bạn là một trợ lý ảo thân thiện và hữu ích."
            )},
            {"role": "user", "content": question}
        ],
        temperature=0.0,
        max_tokens=2048,
        top_p=0.9,
        # frequency_penalty=1.0, # These parameters might not be supported by all models/providers
        # presence_penalty=0.0   # Check OpenRouter documentation for supported parameters
    )
    # Trả về nội dung từ assistant
    return response.choices[0].message.content.strip()

def rewrite(ban_tin):
    if "Tiêu đề" in ban_tin:
       prompt = "Bản tin sau gồm các phần Tiêu đề, Mô tả, Yêu cầu, Quyền lợi (có thể thiếu một vài trong các phần do các phần đó đã đạt yêu cầu) cần được viết lại do nội dung sơ sài, lôm côm. Trong đó phần Tiêu đề và Mô tả đang không liên quan đến nhau. Hãy viết lại các phần sao ngắn gọn, súc tích và rõ nghĩa và tiêu đề và mô tả phải liên quan đến nhau:\n " + ban_tin
    else:
       prompt = "Bản tin sau gồm các phần Mô tả, Yêu cầu, Quyền lợi (có thể thiếu một vài trong các phần do các phần đó đã đạt yêu cầu) cần được viết lại do nội dung sơ sài, lôm côm. Hãy viết lại các phần sao ngắn gọn, súc tích và rõ nghĩa:\n " + ban_tin
    ans = generate_answer(client, prompt)

    return ans

def remove_after_delimiter(text: str, delimiter: str = '---') -> str:
    """
    Trả về phần text trước delimiter (không bao gồm delimiter và nội dung sau).
    Nếu không tìm thấy delimiter, trả về nguyên text.
    """
    parts = text.split(delimiter, 1)
    return parts[0].rstrip()

def extract_job_sections(text: str) -> dict:
    """
    Trả về dict chỉ chứa những key trong 
    ['Tiêu đề', 'Mô tả', 'Yêu cầu', 'Quyền lợi'] thực sự xuất hiện trong text,
    với giá trị tương ứng là nội dung của mỗi phần.
    Cho phép header có thể ở dạng **Tiêu đề:** hoặc Tiêu đề: mà không cần **.
    """
 #   """
 #   pattern = re.compile(
 #       r"""                            # bắt đầu regex
#        (?:\*\*)?                       # có thể có ** ở trước header
#        \s*
#        (Vị\s+trí|Tiêu\s+đề|Mô\s+tả|Yêu\s+cầu|Quyền\s+lợi)  # tên section
 #       \s*:\s*
  #     (?:\*\*)?                       # có thể có ** ngay sau dấu :
  #      \s*
  #      (.*?)                           # nội dung, ít tham lam nhất
 #       (?=                             # lookahead để dừng
 #          (?:\*\*)?\s*                 # có thể có ** ở đầu header kế tiếp
  #         (?:Tiêu\s+đề|Mô\s+tả|Yêu\s+cầu|Quyền\s+lợi)\s*:\s*(?:\*\*)?
  #         |$                           # hoặc đến cuối chuỗi
  #      )
 #       """,
 #       re.DOTALL | re.VERBOSE
 #   )
  #  """
 #   pattern = re.compile(
 #   r"""                            # begin regex
 #   (?:\*\*)?                       # có thể có ** ở trước header
 #   \s*
 #   (Tiêu\s+đề|Mô\s+tả|Yêu\s+cầu|Quyền\s+lợi)  # tên section
  #  (?:                             # tùy chọn ** hoặc :** ngay sau header
  #      \s*\*\*                     #   — ** ngay sau header
 #     | \s*:\s*\*\*?                #   — hoặc : có thể kèm **
 #   )?                              # nhóm trên là tùy chọn
  #  \s*
 #   (.*?)                           # nội dung, ít tham lam nhất
 #   (?=                             # lookahead để dừng trước header kế tiếp
 #      (?:\*\*)?\s*                 #   có thể có ** ở đầu header kế tiếp
  #     (?:Tiêu\s+đề|Mô\s+tả|Yêu\s+cầu|Quyền\s+lợi)
  #     (?:\s*\*\*|\s*:\s*\*\*?)?    #   tiếp tục cho ** hoặc :** sau header
  #     \s*
  #     |$                           #   hoặc đến cuối chuỗi
  #  )
  #  """,
 #   re.DOTALL | re.VERBOSE
 #   )


    pattern = re.compile(r"""
    (?P<section>                                    # tên section
        Tiêu\s+đề
      | Mô\s+tả\s+công\s+việc
      | Mô\s+tả
      | Yêu\s+cầu\s+ứng\s+viên
      | Yêu\s+cầu
      | Quyền\s+lợi
    )
    (?:\s*\*\*|\s*:\s*\*\*?)?                       # tùy chọn ** hoặc :**
    \s*
    (?P<content>.*?)                                # nội dung ít tham lam
    (?=                                             # lookahead dừng trước section kế tiếp
       (?:Tiêu\s+đề
         | Mô\s+tả\s+công\s+việc
         | Mô\s+tả
         | Yêu\s+cầu\s+ứng\s+viên
         | Yêu\s+cầu
         | Quyền\s+lợi
       )
       (?:\s*\*\*|\s*:\s*\*\*?)?                   # ** hoặc :** tiếp theo
       \s*|$                                       # hoặc hết chuỗi
    )
""", re.VERBOSE | re.DOTALL | re.IGNORECASE)


    matches = pattern.findall(text)
    if not matches:
        raise ValueError("Không tìm thấy bất kỳ phần Tiêu đề/Mô tả/Yêu cầu/Quyền lợi nào trong văn bản.")

    result = {}
    print(matches)
    for section_name, content in matches:
        fix_text = remove_after_delimiter(content.strip())
        result[section_name] = re.sub(r'[^\\/\w\s,.;<>]{1,}', '', fix_text)
# key cuối cùng
    last_key = next(reversed(result))
# value tương ứng
    result[last_key] = remove_after_delimiter(result[last_key], "\n\n") 
    #+ "<br />" #Tí xem lại

    return result

def rewrite(ban_tin):
    if "Tiêu đề" in ban_tin:
       prompt = "Bản tin sau gồm các phần Tiêu đề, Mô tả, Yêu cầu, Quyền lợi (có thể thiếu một vài trong các phần do các phần đó đã đạt yêu cầu) cần được viết lại do nội dung sơ sài, lôm côm. Trong đó phần Tiêu đề và Mô tả đang không liên quan đến nhau. Hãy viết lại các phần sao ngắn gọn, súc tích và rõ nghĩa, tiêu đề và mô tả phải liên quan đến nhau và đảm bảo đầy đủ, đúng các thông tin so với bản tin ban đầu:\n " + ban_tin
    else:
       prompt = "Bản tin sau gồm các phần Mô tả, Yêu cầu, Quyền lợi (có thể thiếu một vài trong các phần do các phần đó đã đạt yêu cầu) cần được viết lại do nội dung sơ sài, lôm côm. Hãy viết lại các phần sao ngắn gọn, súc tích, rõ nghĩa và đảm bảo đầy đủ, đúng các thông tin so với bản tin ban đầu:\n " + ban_tin
    ans = generate_answer(client, prompt)
    print(ans)
    result = extract_job_sections(ans)

    return result

def get_ci(d: dict, key: str):
    """
    Lấy value tương ứng với key trong dict d, không phân biệt hoa thường.
    Nếu không tìm thấy, raise KeyError.
    """
    target = key.lower()
    for k, v in d.items():
        if k[0].isalpha() and k[0].isupper():
           if k.lower() == target:
              return v
    return ""