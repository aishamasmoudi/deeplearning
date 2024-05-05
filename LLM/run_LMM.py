from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from datasets import load_dataset

dataset = load_dataset("nedjmaou/MLMA_hate_speech")

llm = Ollama(
    model="llama3", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)
#llm.invoke("The first man on the moon was ...")
print(dataset.head(10))

