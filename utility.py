import nltk
from itertools import chain

punct = set(u''':!),.:;?.]}¢'"、。〉》」』〕〗〞︰︱︳﹐､﹒
﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､￠
々‖•·ˇˉ―′’”([{£¥'"‵〈《「『〔〖（［｛￡￥〝︵︷︹︻
︽︿﹁﹃﹙﹛﹝（｛“‘_…/''')

stemmer = nltk.stem.PorterStemmer()

def format_name(name):
    if name is None:
        return ""
    x = [k.strip() for k in name.lower().strip().replace(".", " ").replace("-", " ").split()]
    return "_".join(x)

def format_symbol(text):
    for token in punct:
        text = text.replace(token, "")
    return text

def format_text(text):
    words = format_symbol(text).split()
    for i, k in enumerate(words):
        words[i] = stemmer.stem(k)
    return words


def transform_feature(key, value):
    if isinstance(value, str):
        value = value.split()

    for i, k in enumerate(value):
        value[i] = "%s:%s" % (key.upper(), k)
    return value

def get_author_feature(paper_id, item):
    author_features = []

    if len(item["authors"]) > 30:
        print(paper_id, len(item["authors"]))
    if len(item["authors"]) > 100:
        return author_features

    title_features = transform_feature("title", format_text(item["title"]))
    keywords_features = transform_feature("keywords", [format_name(k) for k in item.get("keywords", [])])
    venue_features = transform_feature("venue", format_symbol(item.get("venue", "")))

    for i, author in enumerate(item["authors"]):
        name_feature = []
        org_features = []

        org_name = format_name(author.get("org", ""))
        if len(org_name) > 2:
            org_features.extend(transform_feature("org", org_name))

        for j, coauthor in enumerate(item["authors"]):
            if i == j:
                continue
            coauthor_name = coauthor.get("name", "")
            coauthor_org = format_name(coauthor.get("org", ""))
            if len(coauthor_name) > 2:
                name_feature.extend(
                    transform_feature("name", [format_name(coauthor_name)])
                )
            if len(coauthor_org) > 2:
                org_features.extend(
                    transform_feature("org", format_text(coauthor_org.lower()))
                )
        author_features.append(
            name_feature + org_features + title_features + keywords_features + venue_features
        )
    return author_features