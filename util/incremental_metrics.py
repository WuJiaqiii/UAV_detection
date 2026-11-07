"""Validation-only selection metrics; detection misses remain in denominators."""
def incremental_validation_metrics(classes, old_count, totals, correct):
    recalls = {}
    for name, label in classes.items():
        count = totals.get(label, 0)
        if count < 1:
            raise ValueError("Incremental validation requires every active class: " + name)
        if not 0 <= correct.get(label, 0) <= count:
            raise ValueError("Invalid validation counts for " + name)
        recalls[label] = correct.get(label, 0)/count
    old = [value for label, value in recalls.items() if label < old_count]
    new = [value for label, value in recalls.items() if label >= old_count]
    if not old or not new:
        raise ValueError("Old/new harmonic selection requires both class cohorts")
    a, b = sum(old)/len(old), sum(new)/len(new)
    return {"old_macro_recall": 100*a, "new_macro_recall": 100*b,
            "old_new_hmean": 200*a*b/(a+b) if a+b else 0.,
            "record_macro_recall": 100*sum(recalls.values())/len(recalls),
            "record_accuracy": 100*sum(correct.values())/sum(totals.values()),
            "per_class": {name: {"targets": totals[label], "correct": correct.get(label, 0),
                                   "recall": 100*recalls[label]} for name, label in classes.items()}}
