import argparse
import pipeline as P

DEFAULT_DATA = r"C:\Александр\PycharmProjects\uir-service\data\full_df_sector.csv"
DEFAULT_ARTIFACT = r"C:\Александр\PycharmProjects\uir-service\uir_service_artifact"


def render_card(r, manifest):
    """Карточка: распределение вероятностей + ведущее направление как справка."""
    bar = "=" * 60
    lines = [bar,
             f"  Направление (справка) - {r['ticker']}",
             f"  Дата запроса:   {r['date']}",
             f"  Горизонт:       {r['horizon']} торговых дней",
             "-" * 60,
             "  Вероятности классов:"]
    for label, p in sorted(r["probs"].items(), key=lambda kv: -kv[1]):
        marker = "  <- ведущий" if label == r["pred_label"] else ""
        lines.append(f"     {label:<9} {p:.2f}{marker}")
    lines += ["-" * 60,
              f"  Ведущее направление: {r['pred_label'].upper()}  (p={r['lead_proba']:.2f})"]
    if r["low_liquidity"]:
        lines.append(f"  низкая ликвидность ({r['ticker']}) — оценка менее надёжна")
    ctx = r.get("context") or {}
    if ctx:
        lines.append("  Контекст рынка на дату:")
        for k, v in ctx.items():
            lines.append(f"     {k:<20} {v}")
    lines += ["  -- для демонстрации (модель этого не видела) --",
              f"  Факт за +{r['horizon']} дн.: {r['actual_return']*100:+.1f}%  ->  {r['actual_label'].upper()}"
              + ("   (совпало)" if r["hit"] else ""),
              "-" * 60,
              "  справочно: направление носит ознакомительный характер",
              bar]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Справка о направлении по одной бумаге на одну дату.")
    ap.add_argument("--ticker", required=True, help="тикер, например LKOH")
    ap.add_argument("--date", required=True, help="дата YYYY-MM-DD (внутри тестового периода)")
    ap.add_argument("--data", default=DEFAULT_DATA, help="путь к CSV с данными")
    ap.add_argument("--artifact", default=DEFAULT_ARTIFACT, help="папка с сохранённым артефактом")
    args = ap.parse_args()

    manifest, features, scaler, models = P.load_artifact(args.artifact)

    ticker = args.ticker.upper()
    if ticker not in manifest["tickers"]:
        print(f"Неизвестный тикер {ticker}. Доступны: {', '.join(manifest['tickers'])}")
        return

    X, y_ret, ds, ts = P.build_dataset(args.data, manifest, features, scaler)
    r = P.predict_one(X, y_ret, ds, ts, ticker, args.date, manifest, models)

    if r is None:
        print(f"Для {ticker} нет окна на дату {args.date}.")
        print(f"Допустимый период replay: {manifest['test_date_min']} … {manifest['test_date_max']}")
        dates = P.available_dates(ds, ts, ticker)
        near = [d for d in dates if d >= args.date][:3] or dates[-3:]
        if near:
            print("Ближайшие доступные даты:", ", ".join(near))
        return

    r["context"] = P.raw_context_for(args.data, manifest, ticker, args.date)
    print(render_card(r, manifest))


if __name__ == "__main__":
    main()
#