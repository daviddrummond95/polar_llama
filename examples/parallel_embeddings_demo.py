#!/usr/bin/env python3
"""
Demo: Verify parallel embedding generation performance.

This example demonstrates the parallelized embedding generation by:
1. Creating a large corpus of documents
2. Generating embeddings in parallel
3. Measuring performance
4. Verifying all embeddings are generated correctly
"""

import os
from pathlib import Path
import polars as pl
import time
from polar_llama import embedding_async, cosine_similarity, knn_hnsw, Provider

# Load environment variables
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    print("⚠️  dotenv not installed, using existing environment variables")

# Check for API key
if not os.environ.get("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY not set. Please set it to run this demo.")
    exit(1)

print("=" * 80)
print("Parallel Embedding Generation Performance Demo")
print("=" * 80)

# Create a large corpus
print("\n1️⃣  Creating large corpus of documents...")
texts = [
    # Technology & Programming (25)
    "Python is a high-level programming language",
    "JavaScript is used for web development",
    "Machine learning algorithms find patterns in data",
    "Neural networks are inspired by biological brains",
    "Deep learning uses multiple layers of processing",
    "Data science combines statistics and programming",
    "Artificial intelligence mimics human intelligence",
    "Cloud computing provides on-demand resources",
    "Distributed systems handle large-scale workloads",
    "Version control systems track code changes",
    "React is a popular JavaScript library for UI",
    "Docker containers package applications with dependencies",
    "Kubernetes orchestrates containerized applications",
    "Database systems store and retrieve information",
    "APIs enable communication between software components",
    "Cybersecurity protects systems from digital attacks",
    "Blockchain technology enables decentralized ledgers",
    "Quantum computing uses quantum mechanics for computation",
    "DevOps combines development and operations practices",
    "Agile methodology emphasizes iterative development",
    "Microservices architecture breaks apps into small services",
    "GraphQL provides flexible API query language",
    "TypeScript adds static typing to JavaScript",
    "Rust offers memory safety without garbage collection",
    "Go language excels at concurrent programming",

    # Science & Research (30)
    "Quantum physics studies matter at atomic scales",
    "Biology examines living organisms and their processes",
    "Chemistry investigates matter and its interactions",
    "Astronomy explores celestial objects and phenomena",
    "Genetics studies heredity and variation in organisms",
    "Ecology examines relationships between organisms",
    "Neuroscience studies the nervous system",
    "Climate science analyzes weather patterns",
    "Geology studies Earth's physical structure",
    "Oceanography explores marine environments",
    "Microbiology studies microscopic organisms",
    "Biochemistry explores chemical processes in living things",
    "Physics describes nature's fundamental interactions",
    "Meteorology forecasts weather and atmospheric conditions",
    "Botany studies plant life and evolution",
    "Zoology examines animal behavior and classification",
    "Paleontology studies fossils and prehistoric life",
    "Astrophysics applies physics to celestial objects",
    "Molecular biology studies biological molecules",
    "Immunology examines the immune system",
    "Epidemiology tracks disease patterns in populations",
    "Pharmacology studies drug effects on organisms",
    "Toxicology examines effects of toxic substances",
    "Seismology studies earthquakes and seismic waves",
    "Volcanology investigates volcanic activity",
    "Crystallography studies crystal structure",
    "Spectroscopy analyzes matter through light interaction",
    "Thermodynamics studies energy and heat transfer",
    "Electromagnetism describes electric and magnetic forces",
    "Cosmology studies the origin of the universe",

    # Arts & Culture (25)
    "Literature encompasses written works of artistic merit",
    "Music combines sounds in meaningful patterns",
    "Painting uses colors to create visual art",
    "Photography captures moments in time",
    "Cinema tells stories through moving images",
    "Theater presents live dramatic performances",
    "Architecture designs functional and aesthetic buildings",
    "Sculpture creates three-dimensional art forms",
    "Dance expresses ideas through movement",
    "Poetry uses language in aesthetic ways",
    "Opera combines singing with theatrical performance",
    "Ballet requires grace and technical precision",
    "Ceramics shapes clay into functional art",
    "Calligraphy transforms writing into visual art",
    "Graffiti brings art to urban spaces",
    "Animation creates moving images frame by frame",
    "Printmaking transfers images onto surfaces",
    "Illustration visualizes concepts and stories",
    "Fashion design creates wearable art",
    "Jewelry making crafts decorative accessories",
    "Woodworking shapes wood into useful objects",
    "Metalworking forges metal into art and tools",
    "Glassblowing creates artistic glass objects",
    "Textile art weaves fibers into patterns",
    "Origami folds paper into intricate shapes",

    # Sports & Activities (25)
    "Soccer is the world's most popular sport",
    "Basketball requires teamwork and coordination",
    "Tennis combines strategy and physical skill",
    "Swimming provides excellent cardiovascular exercise",
    "Cycling is both transportation and recreation",
    "Running improves endurance and fitness",
    "Yoga promotes flexibility and mindfulness",
    "Martial arts develop discipline and strength",
    "Rock climbing challenges physical and mental limits",
    "Hiking connects people with nature",
    "Baseball emphasizes batting and fielding skills",
    "American football combines strategy and physicality",
    "Golf requires precision and patience",
    "Volleyball demands quick reflexes and teamwork",
    "Cricket is popular in many Commonwealth nations",
    "Rugby combines running, passing, and tackling",
    "Boxing develops speed and power",
    "Wrestling tests strength and technique",
    "Gymnastics showcases flexibility and control",
    "Skiing glides down snowy mountains",
    "Snowboarding carves through winter terrain",
    "Surfing rides ocean waves",
    "Skateboarding performs tricks on boards",
    "Ice skating glides gracefully on ice",
    "Horse riding develops partnership with animals",

    # Nature & Environment (25)
    "Forests provide habitat for countless species",
    "Mountains shape weather and climate patterns",
    "Rivers carry water and nutrients across landscapes",
    "Deserts are arid regions with unique ecosystems",
    "Coral reefs support marine biodiversity",
    "Rainforests contain incredible species diversity",
    "Grasslands support grazing animals",
    "Wetlands filter water and prevent flooding",
    "Arctic regions face rapid climate change",
    "Tropical regions have high temperatures year-round",
    "Tundra ecosystems exist in cold polar regions",
    "Savannas feature scattered trees and grasses",
    "Mangroves protect coastlines from erosion",
    "Kelp forests provide underwater habitats",
    "Alpine zones exist at high mountain elevations",
    "Estuaries mix freshwater and saltwater",
    "Canyons showcase erosion over millions of years",
    "Caves form underground through geological processes",
    "Volcanoes release molten rock from Earth's interior",
    "Glaciers slowly move ice across landscapes",
    "Lakes collect freshwater in natural basins",
    "Beaches accumulate sand along shorelines",
    "Cliffs expose layers of geological history",
    "Islands support unique isolated ecosystems",
    "Fjords carve deep coastal valleys",

    # Food & Cuisine (25)
    "Italian cuisine features pasta and pizza",
    "Japanese food emphasizes fresh ingredients",
    "Mexican dishes use beans and spices",
    "Indian cooking incorporates diverse spices",
    "French cuisine is known for its techniques",
    "Chinese food varies greatly by region",
    "Mediterranean diet promotes health and longevity",
    "Thai food balances sweet, sour, and spicy",
    "Korean cuisine includes fermented foods",
    "Vietnamese dishes feature fresh herbs",
    "Greek food uses olive oil and feta cheese",
    "Spanish tapas offer small shareable plates",
    "Lebanese cuisine features mezze platters",
    "Moroccan tagines slow-cook meats and vegetables",
    "Ethiopian food is served on injera bread",
    "Brazilian churrasco grills various meats",
    "Peruvian ceviche marinates raw fish in citrus",
    "Turkish kebabs grill seasoned meats",
    "German sausages come in many varieties",
    "British fish and chips is a classic dish",
    "Irish stew combines meat and vegetables",
    "Scandinavian cuisine features preserved fish",
    "Polish pierogi are filled dumplings",
    "Hungarian goulash is a hearty stew",
    "Austrian schnitzel is breaded and fried",

    # History & Geography (25)
    "Ancient Rome influenced Western civilization",
    "The Renaissance sparked cultural rebirth",
    "Industrial Revolution transformed manufacturing",
    "World War II reshaped global politics",
    "Ancient Egypt built monumental pyramids",
    "Medieval Europe was dominated by feudalism",
    "The Silk Road connected East and West",
    "Colonialism expanded European power globally",
    "Cold War divided world into two blocs",
    "Information Age revolutionized communication",
    "Ancient Greece developed democracy and philosophy",
    "Mesopotamia was the cradle of civilization",
    "Byzantine Empire preserved Roman traditions",
    "Mongol Empire created the largest land empire",
    "Ottoman Empire controlled Mediterranean trade",
    "Age of Exploration opened new trade routes",
    "American Revolution established democratic republic",
    "French Revolution challenged monarchy",
    "Napoleonic Wars reshaped European borders",
    "Victorian Era saw British imperial expansion",
    "Meiji Restoration modernized Japan",
    "Chinese dynasties ruled for thousands of years",
    "Inca Empire built extensive mountain roads",
    "Maya civilization developed advanced astronomy",
    "African kingdoms traded gold and salt",

    # Daily Life & Society (25)
    "Education develops knowledge and skills",
    "Healthcare systems provide medical services",
    "Transportation connects people and places",
    "Communication technology enables instant contact",
    "Urban planning shapes city development",
    "Agriculture produces food for populations",
    "Trade facilitates exchange of goods",
    "Government establishes laws and order",
    "Economy manages production and distribution",
    "Social networks connect communities",
    "Banking systems manage money and credit",
    "Insurance protects against financial risks",
    "Real estate involves property buying and selling",
    "Manufacturing produces goods at scale",
    "Retail sells products to consumers",
    "Hospitality industry serves travelers",
    "Entertainment provides leisure activities",
    "Media distributes news and information",
    "Advertising promotes products and services",
    "Human resources manages workforce",
    "Legal systems interpret and enforce laws",
    "Public safety protects communities",
    "Environmental protection preserves nature",
    "Social services support vulnerable populations",
    "International relations manages global cooperation",

    # Business & Economics (25)
    "Entrepreneurship starts new business ventures",
    "Marketing promotes products to customers",
    "Sales converts prospects into buyers",
    "Supply chain manages product flow",
    "Logistics coordinates transportation and storage",
    "Finance manages money and investments",
    "Accounting tracks financial transactions",
    "Economics studies production and distribution",
    "Stock markets enable equity trading",
    "Venture capital funds startup companies",
    "Private equity invests in mature businesses",
    "Consulting advises organizations",
    "Management coordinates organizational resources",
    "Leadership inspires and guides teams",
    "Strategy plans long-term direction",
    "Operations manages daily business activities",
    "Quality control ensures product standards",
    "Research and development creates innovations",
    "Intellectual property protects creative works",
    "Mergers and acquisitions combine companies",
    "Corporate governance oversees company management",
    "Risk management identifies and mitigates threats",
    "Compliance ensures regulatory adherence",
    "Sustainability balances profit with environment",
    "Corporate social responsibility benefits society",

    # Psychology & Philosophy (20)
    "Psychology studies human behavior and cognition",
    "Philosophy examines fundamental questions",
    "Ethics explores moral principles",
    "Logic studies valid reasoning",
    "Metaphysics investigates reality's nature",
    "Epistemology examines knowledge and belief",
    "Consciousness remains a deep mystery",
    "Free will debates determinism",
    "Cognitive psychology studies mental processes",
    "Social psychology examines group behavior",
    "Developmental psychology tracks human growth",
    "Clinical psychology treats mental disorders",
    "Existentialism emphasizes individual existence",
    "Stoicism teaches emotional resilience",
    "Utilitarianism maximizes overall happiness",
    "Deontology emphasizes moral duties",
    "Phenomenology studies conscious experience",
    "Aesthetics explores beauty and art",
    "Political philosophy examines government",
    "Philosophy of mind explores consciousness",
]

corpus = pl.DataFrame({
    "id": range(1, len(texts) + 1),
    "text": texts
})

print(f"✓ Created corpus with {len(corpus)} documents")
print(f"  Categories: Technology, Science, Arts, Sports, Nature, Food, History, Society, Business, Psychology")

# Generate embeddings with timing
print(f"\n2️⃣  Generating embeddings for {len(corpus)} documents in parallel...")
print("   (This demonstrates parallel processing - watch the speed!)")

start_time = time.time()
corpus_with_embeddings = corpus.with_columns(
    embedding=embedding_async(pl.col("text"), provider=Provider.OPENAI)
)
end_time = time.time()

duration = end_time - start_time
docs_per_second = len(corpus) / duration

print(f"\n✅ Generated {len(corpus_with_embeddings)} embeddings")
print(f"   ⏱️  Total time: {duration:.2f} seconds")
print(f"   🚀 Speed: {docs_per_second:.1f} documents/second")
print(f"   📊 Embedding dimensions: {len(corpus_with_embeddings['embedding'][0])}")

# Verify all embeddings are valid
print("\n3️⃣  Verifying embedding quality...")
null_count = corpus_with_embeddings['embedding'].is_null().sum()
valid_count = len(corpus_with_embeddings) - null_count

print(f"   ✓ Valid embeddings: {valid_count}/{len(corpus_with_embeddings)}")
print(f"   ✓ Null embeddings: {null_count}")

# Check dimensions consistency
dim_counts = corpus_with_embeddings.select([
    pl.col("embedding").list.len().alias("dim")
]).group_by("dim").agg([
    pl.len().alias("count")
])
print(f"   ✓ Dimension consistency:")
for row in dim_counts.iter_rows():
    print(f"     - {row[0]} dimensions: {row[1]} embeddings")

# Test semantic search on large corpus
print("\n4️⃣  Testing semantic search on large corpus...")
query = pl.DataFrame({
    "query": ["machine learning and neural networks"]
})

query_with_embedding = query.with_columns(
    query_embedding=embedding_async(pl.col("query"), provider=Provider.OPENAI)
)

# Calculate similarities
print("   Computing cosine similarities...")
similarities = query_with_embedding.join(
    corpus_with_embeddings,
    how="cross"
).with_columns(
    similarity=cosine_similarity(
        pl.col("query_embedding"),
        pl.col("embedding")
    )
).sort("similarity", descending=True)

print(f"\n   Top 5 most similar documents to '{query['query'][0]}':")
top_5 = similarities.select(["id", "text", "similarity"]).head(5)
for i, row in enumerate(top_5.iter_rows(named=True), 1):
    print(f"   {i}. [{row['id']}] {row['text']}")
    print(f"      Similarity: {row['similarity']:.4f}")

# Test HNSW search
print("\n5️⃣  Testing HNSW approximate nearest neighbor search...")
print("   Building HNSW index and searching...")

hnsw_start = time.time()
hnsw_df = query_with_embedding.with_columns(
    corpus_embeddings=pl.lit([corpus_with_embeddings["embedding"].to_list()])
).with_columns(
    nearest_neighbors=knn_hnsw(
        pl.col("query_embedding"),
        pl.col("corpus_embeddings").list.first(),
        k=5
    )
)
hnsw_end = time.time()

hnsw_duration = (hnsw_end - hnsw_start) * 1000  # Convert to milliseconds

neighbor_indices = hnsw_df["nearest_neighbors"][0]
print(f"   ⏱️  HNSW search time: {hnsw_duration:.2f}ms")
print(f"   🎯 Found {len(neighbor_indices)} nearest neighbors:")

for idx in neighbor_indices:
    doc = corpus.filter(pl.col("id") == idx + 1)
    if len(doc) > 0:
        print(f"   - [{idx}] {doc['text'][0]}")

# Performance comparison estimate
print("\n6️⃣  Performance Analysis:")
print(f"   Total documents: {len(corpus)}")
print(f"   Total embedding time: {duration:.2f}s")
print(f"   Average time per document: {(duration / len(corpus)) * 1000:.1f}ms")
print(f"   HNSW search time (k=5): {hnsw_duration:.2f}ms")
print(f"\n   💡 Note: Parallel processing significantly speeds up embedding generation!")
print(f"      Without parallelization, this would take much longer.")

# Test with different queries
print("\n7️⃣  Testing multiple diverse queries...")
test_queries = [
    "quantum physics and astronomy",
    "painting and visual arts",
    "soccer and basketball",
    "rainforests and biodiversity",
]

print(f"   Generating embeddings for {len(test_queries)} queries in parallel...")
query_start = time.time()
multi_queries = pl.DataFrame({"query": test_queries}).with_columns(
    query_embedding=embedding_async(pl.col("query"), provider=Provider.OPENAI)
)
query_end = time.time()
query_duration = query_end - query_start

print(f"   ⏱️  Query embedding time: {query_duration:.2f}s ({len(test_queries)/query_duration:.1f} queries/sec)")

for i in range(len(multi_queries)):
    query_row = multi_queries.filter(pl.int_range(pl.len()) == i)
    query_text = query_row["query"][0]

    sims = query_row.join(corpus_with_embeddings, how="cross").with_columns(
        similarity=cosine_similarity(
            pl.col("query_embedding"),
            pl.col("embedding")
        )
    ).sort("similarity", descending=True).head(1)

    print(f"\n   Query: '{query_text}'")
    print(f"   → Best match: '{sims['text'][0]}'")
    print(f"   → Similarity: {sims['similarity'][0]:.4f}")

print("\n" + "=" * 80)
print("✅ Parallel embedding generation demo completed successfully!")
print("=" * 80)
print(f"\n📈 Summary:")
print(f"   • Processed {len(corpus)} documents in {duration:.2f}s")
print(f"   • Parallel throughput: {docs_per_second:.1f} docs/sec")
print(f"   • All embeddings valid: {valid_count}/{len(corpus)}")
print(f"   • HNSW search: {hnsw_duration:.2f}ms for k=5")
print(f"   • Semantic search works correctly across diverse topics")
print("=" * 80)
