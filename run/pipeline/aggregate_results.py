import json, glob, os

rows = []
for d in sorted(glob.glob("run/outputs/*/")):
    f = os.path.join(d, "results.json")
    if not os.path.isfile(f):
        continue
    r = json.load(open(f))
    rows.append(dict(
        name=r.get("model_name"), dims=r.get("spatial_dims"),
        typ=r.get("model_type"), params=r.get("num_parameters", 0),
        dice=r.get("test_mean_dice", 0), acc=r.get("test_pixel_accuracy", 0),
        c1=r.get("test_dice_class_1", 0), c2=r.get("test_dice_class_2", 0),
        c3=r.get("test_dice_class_3", 0),
        best=r.get("best_epoch", 0), t=r.get("training_time_seconds", 0)))

hdr = "{:16}{:10}{:>10}{:>10}{:>9}{:>7}{:>7}{:>7}{:>7}{:>8}".format(
    "model", "type", "params", "meanDice", "pixAcc", "c1", "c2", "c3", "bestEp", "time")
for dims in (2, 3):
    print("\n===== {}D =====".format(dims))
    print(hdr)
    for x in sorted([r for r in rows if r["dims"] == dims], key=lambda z: -z["dice"]):
        print("{:16}{:10}{:8.2f}M{:10.4f}{:9.4f}{:7.3f}{:7.3f}{:7.3f}{:7d}{:6.1f}m".format(
            x["name"], x["typ"], x["params"] / 1e6, x["dice"], x["acc"],
            x["c1"], x["c2"], x["c3"], x["best"], x["t"] / 60))

print("\nTotal runs aggregated: {}/18".format(len(rows)))

# Save a CSV for downstream use
with open("run/outputs/comparison.csv", "w") as out:
    out.write("model,dims,type,params,mean_dice,pixel_acc,dice_c1,dice_c2,dice_c3,best_epoch,time_s\n")
    for x in sorted(rows, key=lambda z: (z["dims"], -z["dice"])):
        out.write("{name},{dims},{typ},{params},{dice:.4f},{acc:.4f},{c1:.4f},{c2:.4f},{c3:.4f},{best},{t:.0f}\n".format(**x))
print("Wrote run/outputs/comparison.csv")
