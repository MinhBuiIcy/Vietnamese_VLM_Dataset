"""Vietnamese prompt builder for VLM dataset generation."""


def build_prompt(has_person: bool) -> str:
    """Return a Vietnamese JSON prompt for captioning and VQA generation.

    Args:
        has_person: Whether the image contains a person.

    Returns:
        Prompt string instructing the model to output structured JSON.
    """
    if has_person:
        activity_instruction = (
            '"activity": "<mô tả hoạt động đang diễn ra trong ảnh, bằng tiếng Việt>"'
        )
    else:
        activity_instruction = '"activity": null  // không có người trong ảnh'

    return f"""Bạn là một trợ lý AI chuyên tạo dữ liệu huấn luyện cho mô hình ngôn ngữ thị giác tiếng Việt.

Hãy phân tích hình ảnh này và trả về **chỉ** một đối tượng JSON hợp lệ theo định dạng sau (không có văn bản nào khác ngoài JSON):

{{
  "caption": "<mô tả chi tiết nội dung hình ảnh bằng tiếng Việt, 2-4 câu>",
  "vqa": [
    {{
      "question": "<câu hỏi bằng tiếng Việt về vật thể hoặc đồ vật trong ảnh>",
      "answer": "<câu trả lời ngắn gọn bằng tiếng Việt>"
    }},
    {{
      "question": "<câu hỏi bằng tiếng Việt về vị trí hoặc không gian trong ảnh>",
      "answer": "<câu trả lời ngắn gọn bằng tiếng Việt>"
    }},
    {{
      "question": "<câu hỏi bằng tiếng Việt về màu sắc, trạng thái hoặc đặc điểm>",
      "answer": "<câu trả lời ngắn gọn bằng tiếng Việt>"
    }}
  ],
  {activity_instruction}
}}

Yêu cầu:
- Tất cả văn bản phải bằng tiếng Việt
- caption phải mô tả chi tiết các đồ vật gia dụng, đồ nội thất và bố cục không gian
- Mỗi câu hỏi VQA phải cụ thể và có thể trả lời từ hình ảnh
- Câu trả lời phải ngắn gọn (1-2 câu)
- Chỉ trả về JSON, không có giải thích thêm"""
