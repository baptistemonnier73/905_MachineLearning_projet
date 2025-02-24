def train(name, num_train_epochs, train_dataset_size):
    dataset = load_dataset("tonyassi/celebrity-1000")["train"]

    names = dataset.features["label"].names

    names_images_dataset = defaultdict(list)

    for label, image in zip(dataset["label"], dataset["image"]):
        names_images_dataset[names[label]].append(image_to_bytes(image))

    validation_names_images_dataset = {name: images[:len(images) // 2] for name, images in
                                       names_images_dataset.items()}
    train_names_images_dataset = {name: images[len(images) // 2:] for name, images in names_images_dataset.items()}

    print("validation dataset :")
    display_images(validation_names_images_dataset[name])

    print("train dataset :")
    display_images(train_names_images_dataset[name])

    limited_dataset = dict(islice(train_names_images_dataset.items(), train_dataset_size))
    exemples_train = [
        {"image": Image.open(io.BytesIO(train_image)), "question": "Can you tell me who is the person in the picture ?",
         "multiple_choice_answer": train_name} for train_name, train_images in limited_dataset.items() for train_image
        in
        train_images]

    model_name = model_id.split("/")[-1]

    training_args = TrainingArguments(
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        learning_rate=1e-4,
        weight_decay=0.01,
        logging_steps=25,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=3,
        optim="paged_adamw_8bit",
        bf16=True,
        output_dir=f"./{model_name}-vqav2",
        hub_model_id=f"{model_name}-vqav2",
        report_to="tensorboard",
        remove_unused_columns=False,
        gradient_checkpointing=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=exemples_train,
    )

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    trainer.train(resume_from_checkpoint=False)

name = "Aaron Taylor-Johnson"
num_train_epochs = 2
train_dataset_size = 5

train(name, num_train_epochs, train_dataset_size)