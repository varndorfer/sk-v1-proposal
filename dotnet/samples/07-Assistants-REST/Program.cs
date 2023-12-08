using System.Net.Http;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.AI.ChatCompletion;
using Microsoft.SemanticKernel.Handlebars;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddHealthChecks();
var app = builder.Build();
app.MapHealthChecks("/health");


string OpenAIApiKey = Env.Var("OpenAI:ApiKey")!;
string BingApiKey = Env.Var("Bing:ApiKey")!;
string currentDirectory = Directory.GetCurrentDirectory();

// Initialize the required functions and services for the kernel
IChatCompletion gpt35Turbo = new OpenAIChatCompletion("gpt-3.5-turbo-1106", OpenAIApiKey);
OllamaGeneration ollamaGeneration = new("wizard-math");

// Create plugins
IPlugin mathPlugin = new Plugin(
    name: "Math",
    functions: NativeFunction.GetFunctionsFromObject(new Math())
);

Plugin ollamaGenerationPlugin = new Plugin(
    name: "Math",
    functions: new() {
                SemanticFunction.GetFunctionFromYaml(currentDirectory + "/Plugins/Ollama/Math.prompt.yaml")
    }
);

IPlugin searchPlugin = new Plugin(
    name: "Search",
    functions: NativeFunction.GetFunctionsFromObject(new Search(BingApiKey))
);

// Create a researcher
IPlugin researcher = AssistantKernel.FromConfiguration(
    currentDirectory + "/Assistants/Researcher.agent.yaml",
    aiServices: new() { gpt35Turbo, },
    plugins: new() { searchPlugin }
);

// Create a mathmatician
IPlugin mathmatician = AssistantKernel.FromConfiguration(
    currentDirectory + "/Assistants/Mathmatician.agent.yaml",
    aiServices: new() { gpt35Turbo, ollamaGeneration },
    plugins: new() { mathPlugin }
);

// Create a Project Manager
AssistantKernel projectManager = AssistantKernel.FromConfiguration(
    currentDirectory + "/Assistants/ProjectManager.agent.yaml",
    aiServices: new() { gpt35Turbo },
    plugins: new() { researcher, mathmatician }
);

IThread thread = await projectManager.CreateThreadAsync();

app.MapPost("chat", async (HttpContext context, Query query) =>
    {
        try
        {
            // Get user input
            thread.AddUserMessageAsync(query.Value);

            // Run the thread using the project manager kernel
            var result = await projectManager.RunAsync(thread);

            // Print the results
            var messages = result.GetValue<List<ModelMessage>>();
            var response = new SKResponse { Value = "" };
            foreach (ModelMessage message in messages)
            {
                response.Value += $"{message}\n";
            }
            return Results.Json(response);
        }
        catch (Exception ex)
        {
            return Results.Json(new SKResponse { Value = ex.Message });
        }
    });

app.Run("http://localhost:5001");