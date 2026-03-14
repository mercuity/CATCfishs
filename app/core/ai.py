
import re
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None
_pipe = None

def _init_model():
    """Инициализирует модель и пайплайн (однократно)."""
    global _model, _tokenizer, _pipe
    if _pipe is not None:
        return
    
    torch.random.manual_seed(0)
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    
    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _pipe = pipeline("text-generation", model=_model, tokenizer=_tokenizer)

def generated(full_name: str, position: str, link: str) -> tuple[str, str]:
    """
    Генерирует тему и тело фишингового письма.
    
    Args:
        full_name: ФИО сотрудника.
        position: Должность.
        link: Трекинговая ссылка для вставки.
    
    Returns:
        Кортеж (тема, HTML-тело). При ошибке — пустые строки.
    """
    _init_model()
    
    system_prompt = """Ты генерируешь реалистичные корпоративные письма на русском языке для тестирования ИБ.
Требования:
- Язык: русский, деловой стиль, без ошибок.
- Формат: чистый HTML-фрагмент (без <html>/<body>).
- Объём: 80–150 слов.
- Персонализация: естественно впиши ФИО и должность.
- Ссылка: вставь предоставленную ссылку в CTA-элемент (кнопка или текст).
- Тема: краткая, нейтральная (5–10 слов).
Запрещено: капс, эмодзи, угрозы, упоминание внешних сервисов."""

    user_content = f"""Получатель: {full_name}, {position}
Ссылка: {link}

Сгенерируй письмо в формате:
THEME: <тема>
---
HTML:
<содержимое>"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.7,
        "do_sample": True,
        "top_p": 0.9,
        "repetition_penalty": 1.15,
    }
    
    output = _pipe(messages, **generation_args)
    raw_text = output[0]['generated_text']

    theme_match = re.search(r'THEME:\s*(.+?)(?=\n---|\nHTML:|$)', raw_text, re.I)
    html_match = re.search(r'HTML:\s*(.+)$', raw_text, re.I | re.DOTALL)
    
    theme = theme_match.group(1).strip() if theme_match else "Уведомление"
    html = html_match.group(1).strip() if html_match else raw_text.strip()
    

    if link not in html:

        if "<p>" in html:
            html = html.replace("<p>", f'<p><a href="{link}" style="color:#1a73e8">Перейти</a></p><p>', 1)
        else:
            html += f'<p><a href="{link}">Перейти</a></p>'
    
    return theme, html