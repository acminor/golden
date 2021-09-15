//
// Created by acminor on 9/7/21.
//

#ifndef GOLDEN_CONVERTERS_HPP
#define GOLDEN_CONVERTERS_HPP

#include <type_traits>

// Adapted from https://en.cppreference.com/w/cpp/types/void_t
#define RegisterHasMethod(METHOD)                                                                                      \
    template <typename, typename = void>                                                                               \
    struct Has##METHOD##_t : std::false_type                                                                           \
    {                                                                                                                  \
    };                                                                                                                 \
    template <typename T>                                                                                              \
    struct Has##METHOD##_t<T, std::void_t<decltype(((T *)0)->METHOD<int, int, int>)>> : std::true_type                 \
    {                                                                                                                  \
    };                                                                                                                 \
    template <typename T>                                                                                              \
    constexpr bool Has##METHOD = Has##METHOD##_t<T>::value;

// RegisterHasMethod(SerializeBase);
// RegisterHasMethod(DeserializeBase);

#include <iostream>

template <typename CRTP>
class IConverter
{
  public:
    template <typename HostType, typename SerialType, typename OptionsType>
    void Serialize(HostType &&a, SerialType &&b, OptionsType c)
    {
        static_cast<CRTP *>(this)->SerializeBase(std::forward<HostType>(a), std::forward<SerialType>(b), c);
    }

    template <typename HostType, typename SerialType, typename OptionsType>
    void Deserialize(HostType &&a, SerialType &&b, OptionsType c)
    {
        static_cast<CRTP *>(this)->DeserializeBase(std::forward<HostType>(a), std::forward<SerialType>(b), c);
    }
};

template <typename SerialFunction, typename DeserialFunction>
class LambdaConverter : IConverter<LambdaConverter<SerialFunction, DeserialFunction>>
{
  public:
    LambdaConverter(SerialFunction serialFunction, DeserialFunction deserialFunction)
        : m_serialFunction(serialFunction), m_deserialFunction(deserialFunction)
    {
    }

    template <typename HostType, typename SerialType, typename ConvertOptions>
    void SerializationBase(HostType &&in, SerialType &&out, ConvertOptions options)
    {
        return m_serialFunction(std::forward<HostType>(in), std::forward<SerialType>(out), options);
    }

    template <typename HostType, typename SerialType, typename ConvertOptions>
    void DeserializationBase(HostType &&out, SerialType &&in, ConvertOptions options)
    {
        return m_deserialFunction(std::forward<HostType>(out), std::forward<SerialType>(in), options);
    }

  private:
    SerialFunction m_serialFunction;
    DeserialFunction m_deserialFunction;
};

template <typename Converter, typename ConverterOptions>
class FilledConverter
{
  public:
    FilledConverter(Converter converter, ConverterOptions options) : m_converter(converter), m_converterOptions(options)
    {
    }

    template <typename HostType, typename SerialType>
    void Serialize(HostType &a, SerialType &b)
    {
        m_converter.Serialize(a, b, m_converterOptions);
    }

    template <typename HostType, typename SerialType>
    void Deserialize(HostType &a, SerialType &b)
    {
        m_converter.Deserialize(a, b, m_converterOptions);
    }

  private:
    Converter m_converter;
    ConverterOptions m_converterOptions;
};

class MultiConverter
{
    template <typename MultiConverterChain, typename FilledConverter, typename ConversionType>
    class MultiConverterBase
    {
      public:
        MultiConverterBase(MultiConverterChain chain) : m_converter(chain)
        {
        }

        template <typename... FilledConverterBases>
        MultiConverterBase(MultiConverterChain chain, FilledConverter a, FilledConverterBases... converters)
            : m_chain(chain), m_converter(a)
        {
        }

        template <typename HostType, typename SerialType>
        void Serialize(HostType a, SerialType b)
        {
            ConversionType temp;
            m_converter.Serialize(a, &temp);
            m_chain.Serialize(temp, &b);
        }

        template <typename HostType, typename SerialType>
        void Deserialize(HostType a, SerialType b)
        {
            m_converter.Deserialize(a, b);
        }

      private:
        MultiConverterChain m_chain;
        FilledConverter m_converter;
    };

  public:
    MultiConverter()
    {
    }

    void SerializeBase()
    {
    }
};

struct ConvertOptions
{
    using Tag = std::nullptr_t;
};

enum class CudaMemoryOptions
{
    Host = 0,
    Device = 1,
    Symbol = 2,
};

template <CudaMemoryOptions MemoryOptionIn>
struct CudaConvertOptions : ConvertOptions
{
  public:
    using Tag = CudaConvertOptions<CudaMemoryOptions::Host>;
    static constexpr CudaMemoryOptions MemoryOption = MemoryOptionIn;
};

template <typename ConvertOptions>
static constexpr bool IsCudaConvertOptions =
    std::is_same_v<CudaConvertOptions<CudaMemoryOptions::Host>, typename ConvertOptions::Tag>;

class ExampleConverter : public IConverter<ExampleConverter>
{
  public:
    template <typename HostType, typename SerialType, typename ConvertOptions>
    void SerializeBase(HostType, SerialType, ConvertOptions)
    {
        static_assert(IsCudaConvertOptions<ConvertOptions>, "Options must be of type CudaConvertOptions");

        if constexpr (ConvertOptions::MemoryOption == CudaMemoryOptions::Host)
            throw 0UL;
        else if constexpr (ConvertOptions::MemoryOption == CudaMemoryOptions::Device)
            throw 1.0f;

        static_assert(ConvertOptions::MemoryOption != CudaMemoryOptions::Symbol, "compile fail");
    }

    template <typename HostType, typename SerialType, typename ConvertOptions>
    void DeserializeBase(HostType, SerialType, ConvertOptions)
    {
    }
};

#endif // GOLDEN_CONVERTERS_HPP
