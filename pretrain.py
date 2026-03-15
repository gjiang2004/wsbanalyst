import json
import random
import re

BOT_PATTERNS = re.compile(
    r"i am a bot|this action was performed automatically|"
    r"contact the moderators|remindme!|ban bet created|ban bet lost|"
    r"bagholder spotted|our ai tracks",
    re.IGNORECASE,
)
URL_ONLY   = re.compile(r"^https?://\S+$")
IMAGE_LINK = re.compile(r"preview\.redd\.it|i\.redd\.it|imgur\.com")
EMOJI_RE   = re.compile(r"[\U00010000-\U0010ffff]")


def is_good_text(text: str, min_len: int = 20, max_len: int = 1200) -> bool:
    if not text:
        return False
    t = text.strip()
    if not (min_len < len(t) < max_len):
        return False
    if BOT_PATTERNS.search(t):
        return False
    if URL_ONLY.match(t):
        return False
    if IMAGE_LINK.search(t):
        return False
    if len(EMOJI_RE.findall(t)) / max(len(t), 1) > 0.20:
        return False
    return True


def clean_text(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def build_instruction(title: str, text: str) -> str:
    """Combine post title and body into a single instruction if body adds context."""
    title = clean_text(title)
    text  = clean_text(text)
    if text and is_good_text(text):
        return f"{title}\n{text}"
    return title


def build_comment_tree(comments: list[dict]) -> dict[str, list[dict]]:
    """Map each comment_id to its direct children as full comment dicts."""
    children: dict[str, list[dict]] = {}
    for c in comments:
        parent_id = c.get("parent_id", "")
        if parent_id.startswith("t1_"):
            pid = parent_id[3:]
            children.setdefault(pid, []).append(c)
    return children


def format_pair(instruction: str, response: str) -> str:
    return f"<s>[INST] {instruction.strip()} [/INST] {response.strip()}</s>"


def build_training_examples(posts: list) -> list[dict]:
    examples: list[str] = []

    for post in posts:
        title    = post.get("title", "")
        text     = post.get("text", "")
        comments = post.get("comments", [])
        if not comments:
            continue

        instruction   = build_instruction(title, text)
        comment_map   = {c["comment_id"]: c for c in comments}
        children      = build_comment_tree(comments)

        if not is_good_text(instruction):
            continue

        # all top-level comments (direct replies to the post)
        top_level = [
            c for c in comments
            if c.get("parent_id", "").startswith("t3_")
            and is_good_text(c.get("body", ""))
        ]

        # sort by score so the best responses come first, but keep all of them
        top_level.sort(key=lambda c: c.get("score", 0), reverse=True)

        # generate one training pair per top-level comment — the model sees
        # the full range of how WSB responds to the same post
        for comment in top_level:
            body = clean_text(comment["body"])
            if is_good_text(body):
                examples.append(format_pair(instruction, body))

        # comment → best direct reply (one pair per parent)
        used_parents: set[str] = set()
        for comment in sorted(comments, key=lambda c: c.get("score", 0), reverse=True):
            parent_id = comment.get("parent_id", "")
            if not parent_id.startswith("t1_"):
                continue
            parent_comment_id = parent_id[3:]
            if parent_comment_id in used_parents:
                continue

            parent = comment_map.get(parent_comment_id)
            if not parent:
                continue

            q = clean_text(parent.get("body", ""))
            if not is_good_text(q):
                used_parents.add(parent_comment_id)
                continue

            # best reply to this parent by score
            siblings   = children.get(parent_comment_id, [])
            best_reply = max(siblings, key=lambda c: c.get("score", 0))
            a          = clean_text(best_reply.get("body", ""))

            if is_good_text(a):
                examples.append(format_pair(q, a))

            used_parents.add(parent_comment_id)

    examples = list(set(examples))
    random.shuffle(examples)
    return [{"text": e} for e in examples]


if __name__ == "__main__":
    with open("wsb_posts.json") as f:
        posts = json.load(f)
    print(f"Loaded {len(posts)} posts")

    examples = build_training_examples(posts)

    short = sum(1 for e in examples if len(e["text"]) < 200)
    avg   = sum(len(e["text"]) for e in examples) // max(len(examples), 1)
    print(f"  total  : {len(examples)}")
    print(f"  short  : {short}  long: {len(examples) - short}")
    print(f"  avg    : {avg} chars")

    with open("wsb_training_data.jsonl", "w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")

    print("✅ Done → wsb_training_data.jsonl")