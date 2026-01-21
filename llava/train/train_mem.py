from llava.train.train import train

if __name__ == "__main__":
    try:
        import os
        from setproctitle import setproctitle
        if os.environ.get("PROCESS_TITLE", None):
            setproctitle(os.environ["PROCESS_TITLE"])
    except ImportError:
        pass
    train(attn_implementation="flash_attention_2")
