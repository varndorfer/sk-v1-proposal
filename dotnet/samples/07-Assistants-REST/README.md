# Assistants

This sample demonstrates how to use Semantic Kernel with assistants.

## Prerequisites

- [.NET 6](https://dotnet.microsoft.com/download/dotnet/6.0) is required to run this sample.
- Install the recommended extensions
- [C#](https://marketplace.visualstudio.com/items?itemName=ms-dotnettools.csharp)
- [Semantic Kernel Tools](https://marketplace.visualstudio.com/items?itemName=ms-semantic-kernel.semantic-kernel) (optional)

## Configuring the sample

Configure an Azure OpenAI endpoint

```
cd ./dotnet/samples/06-Assistants

dotnet user-secrets set "OpenAI:ApiKey" "... your OpenAI key ..."
dotnet user-secrets set ""Bing:ApiKey" "... your Bing key ..."
```

## Running the sample

```
cd ./dotnet/samples/06-Assistants

dotnet run
```



If you start with $25,000 in the stock market and leave it to grow for 20 years at a 5% interest rate, how much would you have?
Expand the following expression: 7(3y+2)