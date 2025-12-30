def collate_return(x):
    x = x[0]
    x["image"] = x["image"].unsqueeze(0)
    if x.get("label") is not None:
        x["label"] = x["label"].unsqueeze(0)
    return x
