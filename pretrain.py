import json
import random
import re

BOT_PATTERNS = re.compile(
    r"i am a bot|this action was performed automatically|"
    r"contact the moderators|remindme!|ban bet created|ban bet lost|"
    r"bagholder spotted|our ai tracks|"
    r"if this bot helps|find me on github|inference station|"
    r"wink.?lab|br.?analysis|br.?fundamentals|"
    r"give it upvotes|not optimized for reddit|"
    r"\[i am a bot",
    re.IGNORECASE,
)
URL_ONLY        = re.compile(r"^https?://\S+$")
IMAGE_LINK      = re.compile(r"preview\.redd\.it|i\.redd\.it|imgur\.com")
EMOJI_RE        = re.compile(r"[\U00010000-\U0010ffff]")
MARKDOWN_LINK   = re.compile(r"\[.*?\]\(https?://")
DELETED_PATTERN = re.compile(r"^\[deleted\]$|^\[removed\]$", re.IGNORECASE)

MAX_SCORE = 50_000
MAX_DEPTH = 4


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
    if MARKDOWN_LINK.search(t):
        return False
    if DELETED_PATTERN.match(t):
        return False
    if len(EMOJI_RE.findall(t)) / max(len(t), 1) > 0.20:
        return False
    return True


def clean_text(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # [label](url) -> keep label
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", "", text)      # [](url) -> remove entirely
    text = re.sub(r"/?u/[A-Za-z0-9_-]+", "", text)         # u/username -> remove entirely, r/subreddit kept
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

        instruction = build_instruction(title, text)
        children    = build_comment_tree(comments)

        if not is_good_text(instruction):
            continue

        # all top-level comments (direct replies to the post)
        top_level = [
            c for c in comments
            if c.get("parent_id", "").startswith("t3_")
            and is_good_text(c.get("body", ""))
            and c.get("score", 0) <= MAX_SCORE
        ]

        # sort by score so the best responses come first, but keep all of them
        top_level.sort(key=lambda c: c.get("score", 0), reverse=True)

        # generate one training pair per top-level comment -- the model sees
        # the full range of how WSB responds to the same post
        for comment in top_level:
            body = clean_text(comment["body"])
            if is_good_text(body):
                examples.append(format_pair(instruction, body))

        # walk reply chains up to MAX_DEPTH levels deep
        # each node becomes a (context, best_reply) training pair
        def walk_chain(comment_id: str, context: str, depth: int) -> None:
            if depth > MAX_DEPTH:
                return
            siblings = [
                c for c in children.get(comment_id, [])
                if c.get("score", 0) <= MAX_SCORE
            ]
            if not siblings:
                return
            best_reply = max(siblings, key=lambda c: c.get("score", 0))
            a = clean_text(best_reply.get("body", ""))
            if is_good_text(a):
                examples.append(format_pair(context, a))
                walk_chain(best_reply["comment_id"], a, depth + 1)

        for comment in top_level:
            body = clean_text(comment["body"])
            if is_good_text(body):
                walk_chain(comment["comment_id"], body, 2)

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

    print("Done -> wsb_training_data.jsonl")