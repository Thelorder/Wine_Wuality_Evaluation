 self.wine_df.columns]

        # Chroma setup - Ephemeral (fast) or switch to Persistent later
        self.client = chromadb.EphemeralClient()
        
        self.embedding_func = embedding_functions.OllamaEmbeddingFunction(
            model_name="nomic-embed-text",
            url="http://localhost:11434/api/embeddings",
            timeout = 300
        )

        self.collection = self.client.get_or_create_collection(
            name="wine_knowledge",
            embedding_function=self.embedding_func,
            metadata={"hnsw:space": "cosine"}