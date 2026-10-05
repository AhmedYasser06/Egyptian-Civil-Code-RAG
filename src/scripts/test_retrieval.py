from src.retrieval.retriever import LegalRetriever


def print_results(
    query: str,
    results,
):

    print()
    print("=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    for i, result in enumerate(
        results,
        start=1,
    ):

        print()
        print(f"#{i}")

        print(
            f"Score: "
            f"{result['score']:.4f}"
        )

        print(
            f"Article: "
            f"{result['article_number']}"
        )

        print(
            f"Language: "
            f"{result['language']}"
        )

        print(
            f"Citation: "
            f"{result['citation']}"
        )

        print(
            f"Topic: "
            f"{result['topic']}"
        )

        print(
            f"Text:\n"
            f"{result['text'][:500]}"
        )


def main():

    retriever = LegalRetriever()

    arabic_query = (
        "ما هي الحالات التي يكون فيها "
        "استعمال الحق غير مشروع؟"
    )

    arabic_results = retriever.retrieve(
        arabic_query,
        top_k=5,
    )

    print_results(
        arabic_query,
        arabic_results,
    )

    english_query = (
        "When is the exercise of a right "
        "considered unlawful?"
    )

    english_results = retriever.retrieve(
        english_query,
        top_k=5,
    )

    print_results(
        english_query,
        english_results,
    )


if __name__ == "__main__":
    main()