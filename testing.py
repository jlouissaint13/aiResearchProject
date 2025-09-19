import sys

from flask import Flask
import pymupdf4llm
import pathlib
from ChunkSplitter import ChunkSplitter
from ChromaManager import ChromaManager
from EmbedHandler import EmbedHandler
from Rag import Rag
chroma = ChromaManager()
chunkSplitter = ChunkSplitter()
embedHandler = EmbedHandler()


text = """
Artificial Intelligence (AI) has rapidly become one of the most transformative technologies in the modern era. Over the last decade, the integration of AI into our everyday lives has accelerated at an unprecedented pace. From the rise of digital assistants like Siri and Alexa to advanced recommendation systems on platforms such as Netflix, YouTube, and Amazon, AI is shaping how we interact with technology, consume information, and make decisions. Yet the story of AI is not just about technological progress—it is also about social impact, ethical concerns, and the reshaping of economies and labor markets.

The origins of AI date back to the mid-20th century, when pioneering computer scientists such as Alan Turing began to ask whether machines could think. The famous “Turing Test” was proposed as a way to evaluate whether a machine could convincingly mimic human intelligence in conversation. In the decades that followed, AI research went through cycles of intense optimism and periods of stagnation known as “AI winters.” During these winters, funding dried up, progress slowed, and skepticism grew about the feasibility of building truly intelligent systems. However, the arrival of modern machine learning, powered by vast computational resources and immense datasets, reignited AI research and fueled today’s renaissance.

At the heart of AI is machine learning, the ability of systems to improve performance by learning from data rather than relying solely on explicit programming. In supervised learning, algorithms are trained on labeled examples, enabling them to classify new data or make predictions. In unsupervised learning, algorithms uncover hidden patterns without explicit labels. Reinforcement learning, another major paradigm, trains agents to act within environments by rewarding successful actions and discouraging failures. These techniques, combined with deep learning’s multi-layered neural networks, have propelled AI systems to achieve remarkable feats in vision, speech recognition, translation, and even creativity.

One of the most visible applications of AI has been in the field of natural language processing (NLP). Breakthroughs such as transformers and large language models have enabled machines to generate coherent text, translate across languages, and summarize complex documents. These advancements are not merely technical marvels; they hold the potential to reshape industries ranging from customer service to education and journalism. For example, chatbots and virtual assistants are increasingly used by companies to handle customer inquiries, reducing costs while providing immediate responses. In academia, AI-driven writing tools can assist students with brainstorming, drafting, and refining essays, although concerns remain about over-reliance and academic integrity.

Another critical domain where AI is driving change is healthcare. AI-powered diagnostic tools are helping doctors detect diseases such as cancer, diabetes, and cardiovascular conditions with greater accuracy and speed. Algorithms can analyze medical images to flag anomalies that might otherwise be overlooked. Personalized medicine, where treatments are tailored to individual genetic profiles, is becoming more achievable through AI-driven data analysis. However, the use of AI in healthcare also raises questions about bias, data privacy, and accountability. If an algorithm makes an incorrect diagnosis, who is responsible—the developer, the doctor, or the machine itself?

The rise of AI has also prompted significant debate about the future of work. Automation powered by AI is capable of handling repetitive tasks once reserved for humans, from manufacturing assembly lines to clerical data entry. While this can lead to increased productivity and lower costs for businesses, it also creates concerns about job displacement. Studies have shown that while AI may eliminate certain jobs, it also creates new opportunities in fields such as AI ethics, data annotation, and algorithmic auditing. The challenge lies in ensuring that workers are reskilled and supported during these transitions, so the benefits of AI are distributed equitably across society.

Ethical concerns form a crucial dimension of AI development. Algorithms are only as unbiased as the data they are trained on, and since real-world data often reflects societal inequities, AI can inadvertently perpetuate discrimination. For instance, facial recognition technologies have been criticized for disproportionately higher error rates when identifying people of color. Similarly, predictive policing algorithms may reinforce existing biases in law enforcement practices. Addressing these issues requires a combination of technical fixes, such as bias detection and mitigation strategies, as well as broader policy interventions and public oversight.

Another area of concern is the use of AI in surveillance and national security. Governments and corporations alike are deploying AI to monitor populations, track individuals, and analyze vast amounts of personal data. While proponents argue that such systems enhance safety and efficiency, critics warn that unchecked surveillance could erode privacy and civil liberties. The balance between leveraging AI for security and protecting individual freedoms remains one of the most pressing dilemmas in the digital age.

Despite these challenges, the future of AI is filled with possibilities. In education, AI could help personalize learning experiences by adapting to individual student needs and providing targeted feedback. In environmental science, AI models are already being used to predict climate patterns, optimize energy usage, and monitor deforestation. In creative fields, AI-generated art, music, and literature are pushing the boundaries of what it means to create and innovate. While some worry that AI creativity may undermine human originality, others argue that these tools can serve as collaborators, expanding the horizons of human expression.

The global race for AI leadership has geopolitical implications as well. Nations such as the United States, China, and members of the European Union are investing heavily in AI research and development, recognizing its potential to shape economic competitiveness and military power. This competition has spurred both collaboration and tension, as countries seek to establish standards, secure access to talent, and protect intellectual property. International dialogue around AI governance will be essential to prevent misuse, ensure transparency, and promote responsible innovation.

For individuals, the rise of AI presents both opportunities and responsibilities. On one hand, AI tools can enhance productivity, creativity, and access to knowledge. On the other hand, individuals must cultivate digital literacy to critically evaluate the outputs of AI systems and understand their limitations. Blind trust in algorithmic decisions can lead to harmful consequences, while informed skepticism can help ensure that AI serves human interests rather than undermining them.

In conclusion, Artificial Intelligence is not just another technological trend—it is a paradigm shift with far-reaching consequences for how we live, work, and relate to one another. Its promise is immense, offering solutions to some of humanity’s most pressing challenges. Yet its risks are equally profound, from exacerbating inequality to threatening privacy and autonomy. The path forward requires a balance between innovation and caution, enthusiasm and responsibility. As we continue to integrate AI into the fabric of our societies, we must remain vigilant stewards, ensuring that this powerful tool is used to advance human flourishing rather than diminish it.

The story of AI is, in many ways, a story of ourselves. It reflects our ambitions, our fears, and our values. Whether AI ultimately becomes a force for good or harm will depend not only on technical breakthroughs but also on the ethical choices we make as individuals, organizations, and nations. In this sense, the question of artificial intelligence is inseparable from the broader question of human intelligence: how we choose to define it, cultivate it, and use it for the benefit of all.
"""


def pdf_to_text():
    pdf_path  = input("Please enter the file path of the pdf: ")
    md_text = pymupdf4llm.to_markdown(pdf_path)
    base_directory = pathlib.Path(__file__).parent
    output_directory = base_directory / "pdfToText"

    pathlib.Path(output_directory / "result.md").write_text(md_text,encoding="utf-8")

    return md_text

def test_chunk_embedding(text):
    chunks = chunkSplitter.semantic_split(text)

    embedHandler.store_chunks_and_embeddings(chunks)

def test_database():

    chroma.check_db()

def question():
    q = "y"
    while q == "yes" or q == "y":
        user_question = input("What is your question? ")
        rag = Rag(chroma,user_question)
        print(rag.model_run())
        q = input("Would you like to ask another question?(Y/N) ").lower()

if __name__ == '__main__':
    user_input = -1

    while user_input != 6:
        try:
            print("Please make a selection\n1)Ask a question\n2)Insert a pdf\n3)Check the database\n4)Delete the database\n5)Choose your model(Work In Progress)\n6)Quit")
            user_input = int(input())


            match user_input:
                case 1: question()
                case 2: test_chunk_embedding(pdf_to_text()) #pdf_to_text()
                case 3: chroma.check_db()
                case 4: chroma.delete()
                case 5: sys.exit(1)
                case 6: sys.exit(0)
                case _: print("Invalid selection")
        except ValueError: print("Please Enter A Numerical Value")





