//
// Created by acminor on 9/7/21.
//

#pragma once

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

/*
 * Based off of example implementation of std::remove_extent from
 * https://en.cppreference.com/w/cpp/types/remove_extent
 *
 * For now the only normalization this handles is making an array become
 * a pointer for use with our pointer-based API
 */
template <typename T>
struct NormalizeType
{
    typedef T type;
};

template <typename T>
struct NormalizeType<T[]>
{
    typedef T* type;
};

template <typename T, std::size_t N>
struct NormalizeType<T[N]>
{
    typedef T* type;
};

template <typename T>
struct NormalizeType<T(&)[]>
{
    typedef T* type;
};

template <typename T, std::size_t N>
struct NormalizeType<T(&)[N]>
{
    typedef T* type;
};

template <typename CRTP>
class IConverter
{
  public:
    template <typename HostType, typename SerialType, typename OptionsType>
    void Serialize(HostType &&a, SerialType &&b, OptionsType c)
    {
        using type = typename NormalizeType<HostType>::type;
        static_cast<CRTP *>(this)->SerializeBase(std::forward<type>(a),
            std::forward<SerialType>(b), c);
    }

    template <typename HostType, typename SerialType, typename OptionsType>
    void Deserialize(HostType &&a, SerialType &&b, OptionsType c)
    {
        using type = typename NormalizeType<HostType>::type;
        static_cast<CRTP *>(this)->DeserializeBase(std::forward<type>(a),
            std::forward<SerialType>(b), c);
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
    void Serialize(HostType &&a, SerialType &&b)
    {
        m_converter.Serialize(std::forward<HostType>(a), std::forward<SerialType>(b), m_converterOptions);
    }

    template <typename HostType, typename SerialType>
    void Deserialize(HostType &&a, SerialType &&b)
    {
        m_converter.Deserialize(std::forward<HostType>(a), std::forward<SerialType>(b), m_converterOptions);
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

template <typename HostType, typename Converter>
class ConverterHandle
{
  public:
    explicit ConverterHandle(HostType &hostVariable, Converter &converter)
        : m_hostVariable(hostVariable), m_converter(converter)
    {
    }

    template <typename SerialType>
    ConverterHandle &operator=(const SerialType &serialVariable)
    {
        m_hostVariable = m_converter(serialVariable);
        return *this;
    }

  private:
    HostType &m_hostVariable;
    Converter &m_converter;
};

/**
 * @enum MemoryOptions
 * @brief Describes the memory allocation/copy options for a converter.
 */
enum class MemoryOptions
{
    /**
     * Host memory, no allocation
     * - different to device memory which is assumed to be unallocated
     * - this can be used to handle stack variables
     */
    Host = 0,
    /**
     * Device memory, allocation performed
     * - this will allocate new memory for the device variable
     */
    Device = 1,
    /**
     * Symbol memory, no allocation (constant memory)
     */
    Symbol = 2,
    HostAlloc = 3,
    DeviceNoAlloc = 4,
};

constexpr bool IsAllocOption(MemoryOptions memoryOptions)
{
    return memoryOptions == MemoryOptions::Device || memoryOptions == MemoryOptions::HostAlloc;
}

constexpr bool IsCopyOnlyOption(MemoryOptions memoryOptions)
{
    return !IsAllocOption(memoryOptions);
}

template <typename HostType, MemoryOptions memoryOptions>
class StorageHandle
{
  public:
    explicit StorageHandle(HostType hostVariable) : m_hostVariable(hostVariable)
    {
    }

    template <typename SerialType>
    StorageHandle &operator=(const SerialType &serialVariable)
    {
        m_hostVariable = serialVariable;
        return *this;
    }

  private:
    HostType &m_hostVariable;
};

class IdentityConverter : IConverter<IdentityConverter>
{
  public:
    template <typename HostType, typename SerialType>
    void SerializeBase(HostType &a, SerialType &b)
    {
        b = a;
    }

    template <typename HostType, typename SerialType>
    void DeserializeBase(HostType &a, SerialType &b)
    {
        a = b;
    }

    template <typename HostType, typename SerialType>
    void DeserializeBase(HostType *a, SerialType &b)
    {
        *a = b;
    }

    // TODO extract into IFilledConverter

    template <typename HostType, typename SerialType>
    void Serialize(HostType &&a, SerialType &&b)
    {
        this->DeserializeBase(std::forward<HostType>(a), std::forward<SerialType>(b));
    }

    template <typename HostType, typename SerialType>
    void Deserialize(HostType &&a, SerialType &&b)
    {
        this->DeserializeBase(std::forward<HostType>(a), std::forward<SerialType>(b));
    }
};

template <typename ConverterType>
struct SubConverterOptions
{
    static constexpr char Tag[] = "SubConverterOptions";

    explicit SubConverterOptions(ConverterType converter) : Converter(converter)
    {
    }

    ConverterType Converter;
};

template <MemoryOptions memoryOptions, typename ConverterType>
struct ConvertOptions
{
    static constexpr char Tag[] = "ConvertOptions";

    explicit ConvertOptions(SubConverterOptions<ConverterType> subConverterOptions)
        : SubConverterOpts(subConverterOptions)
    {
    }

    static constexpr MemoryOptions MemoryOption = memoryOptions;
    SubConverterOptions<ConverterType> SubConverterOpts;
};

template <MemoryOptions memoryOptions, MemoryOptions otherMemoryOptions, typename ConverterType>
auto make_options(const ConvertOptions<otherMemoryOptions, ConverterType> &c)
    -> ConvertOptions<memoryOptions, ConverterType>
{
    return ConvertOptions<memoryOptions, ConverterType>(c.SubConverterOpts);
}

template <MemoryOptions memoryOptions>
auto make_options() -> ConvertOptions<memoryOptions, IdentityConverter>
{
    return ConvertOptions<memoryOptions, IdentityConverter>(SubConverterOptions(IdentityConverter()));
}

template <MemoryOptions memoryOptions, typename ConverterType>
auto make_options(ConverterType converter) -> ConvertOptions<memoryOptions, ConverterType>
{
    return ConvertOptions<memoryOptions, ConverterType>(SubConverterOptions<ConverterType>(converter));
}

template <MemoryOptions memoryOptions, typename ConverterType>
auto make_options(SubConverterOptions<ConverterType> converterOpts) -> ConvertOptions<memoryOptions, ConverterType>
{
    return ConvertOptions<memoryOptions, ConverterType>(converterOpts);
}

template <MemoryOptions MemoryOptionIn>
struct CudaConvertOptions : ConvertOptions<MemoryOptionIn, IdentityConverter>
{
  public:
    using Tag = CudaConvertOptions<MemoryOptions::Host>;
    static constexpr MemoryOptions MemoryOption = MemoryOptionIn;
};

template <typename ConvertOptions>
static constexpr bool IsCudaConvertOptions =
    std::is_same_v<CudaConvertOptions<MemoryOptions::Host>, typename ConvertOptions::Tag>;

class ExampleConverter : public IConverter<ExampleConverter>
{
  public:
    template <typename HostType, typename SerialType, typename ConvertOptions>
    void SerializeBase(HostType, SerialType, ConvertOptions)
    {
        static_assert(IsCudaConvertOptions<ConvertOptions>, "Options must be of type CudaConvertOptions");

        if constexpr (ConvertOptions::MemoryOption == MemoryOptions::Host)
            throw 0UL;
        else if constexpr (ConvertOptions::MemoryOption == MemoryOptions::Device)
            throw 1.0f;

        static_assert(ConvertOptions::MemoryOption != MemoryOptions::Symbol, "compile fail");
    }

    template <typename HostType, typename SerialType, typename ConvertOptions>
    void DeserializeBase(HostType, SerialType, ConvertOptions)
    {
    }
};

class ConverterException : public std::exception
{
  public:
    explicit ConverterException(std::string message) : m_message(message)
    {
    }

    const char *what() const noexcept override
    {
        return m_message.c_str();
    }

  private:
    std::string m_message;
};