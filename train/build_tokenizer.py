def save_processor_from_tokenizer(tokenizer_obj: Tokenizer, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    tok_json_path = os.path.join(out_dir, "tokenizer.json")
    tokenizer_obj.save(tok_json_path)
    logger.info("Сохранён tokenizers JSON: %s", tok_json_path)

    tok_fast = PreTrainedTokenizerFast(tokenizer_file=tok_json_path,
                                       unk_token="[UNK]", pad_token="[PAD]",
                                       bos_token="[BOS]", eos_token="[EOS]")

    # Создаём ViTFeatureExtractor и явно сохраняем его конфиг в out_dir
    feat = ViTFeatureExtractor(do_resize=True, size=getattr(config, "TOKENIZER_IMAGE_SIZE", (384, 384)), do_normalize=True)
    try:
        # сохраняем конфиг feature extractor (это создаст preprocessor_config.json / image_processor_config.json)
        feat.save_pretrained(out_dir)
    except Exception as e:
        logger.warning("Не удалось сохранить feature_extractor конфиг: %s", e)

    # Построим процессор из объектов (не из файлов) — это надёжно:
    processor = TrOCRProcessor(feature_extractor=feat, tokenizer=tok_fast)

    # Сохраняем processor и токенайзер (на всякий случай)
    try:
        processor.save_pretrained(out_dir)
    except Exception as e:
        logger.warning("processor.save_pretrained failed: %s", e)
    try:
        tok_fast.save_pretrained(out_dir)
    except Exception as e:
        logger.warning("tok_fast.save_pretrained failed: %s", e)

    logger.info("Processor сохранён в %s", out_dir)
    return processor
