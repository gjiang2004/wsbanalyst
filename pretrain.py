import json
import random

def load_posts(filename="wsb_posts.json"):
    with open(filename, 'r') as f:
        return json.load(f)

def format_conversation(instruction, response):
    return f"<s>[INST] {instruction.strip()} [/INST] {response.strip()}</s>"

def is_good_text(text, min_len=15, max_len=800):
    return text and min_len < len(text.strip()) < max_len

def build_training_examples(posts):
    examples = []

    for post in posts:
        title = post.get('title', '').strip()
        body = post.get('text', '').strip()
        comments = post.get('comments', [])

        good_comments = [c for c in comments if is_good_text(c.get('body', ''))]
        if not good_comments:
            continue

        top_comment = good_comments[0]['body'].strip()

        # 1. Title -> top comment
        if is_good_text(title) and is_good_text(top_comment):
            examples.append(format_conversation(title, top_comment))

        # 2. Title -> body
        if is_good_text(title) and is_good_text(body):
            examples.append(format_conversation(title, body))

        # 3. Body -> top comment
        if is_good_text(body) and is_good_text(top_comment):
            examples.append(format_conversation(body, top_comment))

        # 4. Parent -> best direct reply (max 1 reply per parent)
        comment_map = {c['comment_id']: c for c in comments}
        used_parents = set()

        for comment in good_comments:
            parent_id = comment.get('parent_id', '')
            if not parent_id.startswith('t1_'):
                continue
            parent_comment_id = parent_id[3:]
            if parent_comment_id in used_parents:
                continue
            parent = comment_map.get(parent_comment_id)
            if not parent:
                continue
            q = parent.get('body', '').strip()
            a = comment.get('body', '').strip()
            if is_good_text(q) and is_good_text(a):
                examples.append(format_conversation(q, a))
                used_parents.add(parent_comment_id)

    examples = list(set(examples))
    return [{"text": e} for e in examples]

if __name__ == "__main__":
    print("Loading posts...")
    posts = load_posts("wsb_posts.json")
    print(f"Loaded {len(posts)} posts")

    print("Building conversational training examples...")
    examples = build_training_examples(posts)

    random.shuffle(examples)
    examples = examples[:5000]

    output_file = "wsb_training_data.jsonl"
    with open(output_file, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')

    total = len(examples)
    avg = sum(len(e['text']) for e in examples) // total
    print(f"✅ Done! {total} training examples saved to {output_file}")
    print(f"   Avg length: {avg} chars per example")