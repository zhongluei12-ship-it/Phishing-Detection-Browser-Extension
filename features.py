from bs4 import BeautifulSoup

# ----------------------------
# BOOLEAN FEATURES
# ----------------------------

def has_title(soup):
    return int(bool(soup.title and soup.title.text.strip()))

def has_input(soup):
    return int(bool(soup.find_all("input")))

def has_button(soup):
    return int(bool(soup.find_all("button")))

def has_image(soup):
    return int(bool(soup.find_all("image")))

def has_submit(soup):
    for input_tag in soup.find_all("input"):
        if input_tag.get("type") == "submit":
            return 1
    return 0

def has_link(soup):
    return int(bool(soup.find_all("link")))

def has_password(soup):
    for input_tag in soup.find_all("input"):
        if input_tag.get("type") == "password":
            return 1
    return 0

def has_email_input(soup):
    for input_tag in soup.find_all("input"):
        if input_tag.get("type") == "email":
            return 1
    return 0

def has_hidden_element(soup):
    for input_tag in soup.find_all("input"):
        if input_tag.get("type") == "hidden":
            return 1
    return 0

def has_audio(soup):
    return int(bool(soup.find_all("audio")))

def has_video(soup):
    return int(bool(soup.find_all("video")))

def has_h1(soup):
    return int(bool(soup.find_all("h1")))

def has_h2(soup):
    return int(bool(soup.find_all("h2")))

def has_h3(soup):
    return int(bool(soup.find_all("h3")))

def has_footer(soup):
    return int(bool(soup.find_all("footer")))

def has_form(soup):
    return int(bool(soup.find_all("form")))

def has_text_area(soup):
    return int(bool(soup.find_all("textarea")))

def has_iframe(soup):
    return int(bool(soup.find_all("iframe")))

def has_text_input(soup):
    for input_tag in soup.find_all("input"):
        if input_tag.get("type") == "text":
            return 1
    return 0

def has_nav(soup):
    return int(bool(soup.find_all("nav")))

def has_object(soup):
    return int(bool(soup.find_all("object")))

def has_picture(soup):
    return int(bool(soup.find_all("picture")))

# ----------------------------
# COUNT FEATURES
# ----------------------------

def number_of_inputs(soup):
    return len(soup.find_all("input"))

def number_of_buttons(soup):
    return len(soup.find_all("button"))

def number_of_images(soup):
    count = len(soup.find_all("image"))
    for meta in soup.find_all("meta"):
        if (meta.get("type") or meta.get("name")) == "image":
            count += 1
    return count

def number_of_option(soup):
    return len(soup.find_all("option"))

def number_of_list(soup):
    return len(soup.find_all("li"))

def number_of_TH(soup):
    return len(soup.find_all("th"))

def number_of_TR(soup):
    return len(soup.find_all("tr"))

def number_of_href(soup):
    count = 0
    for link in soup.find_all("link"):
        if link.get("href"):
            count += 1
    return count

def number_of_paragraph(soup):
    return len(soup.find_all("p"))

def number_of_script(soup):
    return len(soup.find_all("script"))

def number_of_clickable_button(soup):
    count = 0
    for button in soup.find_all("button"):
        if button.get("type") == "button":
            count += 1
    return count

def number_of_a(soup):
    return len(soup.find_all("a"))

def number_of_img(soup):
    return len(soup.find_all("img"))

def number_of_div(soup):
    return len(soup.find_all("div"))

def number_of_figure(soup):
    return len(soup.find_all("figure"))

def number_of_meta(soup):
    return len(soup.find_all("meta"))

def number_of_sources(soup):
    return len(soup.find_all("source"))

def number_of_span(soup):
    return len(soup.find_all("span"))

def number_of_table(soup):
    return len(soup.find_all("table"))

# ----------------------------
# LENGTH FEATURES
# ----------------------------

def length_of_title(soup):
    if soup.title:
        return len(soup.title.text.strip())
    return 0

def length_of_text(soup):
    return len(soup.get_text(strip=True))

# ----------------------------
# VECTOR CREATION
# ----------------------------

def create_vector(soup):
    return [
        has_title(soup),
        has_input(soup),
        has_button(soup),
        has_image(soup),
        has_submit(soup),
        has_link(soup),
        has_password(soup),
        has_email_input(soup),
        has_hidden_element(soup),
        has_audio(soup),
        has_video(soup),
        number_of_inputs(soup),
        number_of_buttons(soup),
        number_of_images(soup),
        number_of_option(soup),
        number_of_list(soup),
        number_of_TH(soup),
        number_of_TR(soup),
        number_of_href(soup),
        number_of_paragraph(soup),
        number_of_script(soup),
        length_of_title(soup),
        has_h1(soup),
        has_h2(soup),
        has_h3(soup),
        length_of_text(soup),
        number_of_clickable_button(soup),
        number_of_a(soup),
        number_of_img(soup),
        number_of_div(soup),
        number_of_figure(soup),
        has_footer(soup),
        has_form(soup),
        has_text_area(soup),
        has_iframe(soup),
        has_text_input(soup),
        number_of_meta(soup),
        has_nav(soup),
        has_object(soup),
        has_picture(soup),
        number_of_sources(soup),
        number_of_span(soup),
        number_of_table(soup)
    ]
