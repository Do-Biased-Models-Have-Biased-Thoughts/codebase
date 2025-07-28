## Utilizing HaRiM+ score as a proxy for bias detection in LLM thoughts

The **HaRiM+ Score** evaluates the factual consistency of generated content relative to its source. We repurpose the HaRiM+ Score as a **proxy for bias detection** in **LLM-generated explanations (thoughts)**. Specifically, we compute HaRiM+ scores by comparing the model's explanations against the combination of the **question and context** from the **BBQ dataset**. A **higher HaRiM+ Score** indicates that the LLM's explanation closely aligns with the original question and context, suggesting a **lower likelihood of hallucination** and, consequently, **less bias**. Conversely, a **lower HaRiM+ Score** reflects greater deviation from the source material, signaling a **higher risk of biased reasoning**.

Please run `python harim_plus.py` to collect harim+ score across all the 5 models' test.
