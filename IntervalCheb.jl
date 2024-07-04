using IntervalArithmetic, LinearAlgebra
include("FourierChebyshev.jl")
# 
# Chebyshev points of second kind
function interval_chebpts(n, a=-1, b=1) # n: maximum order of Chebyshev polynomials
    m = Vector(-n:2:n)
    x = sinpi.(interval.(m) / (interval(2) * interval(n)))
    if a == -1 && b == 1
        return x
    else
        return (interval(1.0) .- x) .* interval(a) / interval(2) + (interval(1.0) .+ x) .* interval(b) / interval(2)
    end
end
# 
# Interval version of the FFT
function verifyfft(z::Vector{Complex{Interval{T}}}, sign=1) where {T<:Real}
    n = length(z)
    col = 1
    array1 = true
    if n == 1
        return z
    else
        isrow_ = false
    end
    log2n = Int(round(log2(n))) #check dimension
    if 2^log2n ≠ n # return error if n is not the powers of 2
        error("length must be power of 2")
    end
    #bit-reversal
    f = 2^(log2n - 1)
    v = [0; f]
    for k = 1:log2n-1
        f = f >> 1
        v = append!(v, f .+ v)
    end
    z2 = zeros(Complex{Interval{T}}, n, col)
    # if isa(real(z[1]),Interval)
    #     z2 = map(Interval{T},z2)
    # end
    # replace z
    for j = 1:n
        z2[j, :] = z[v[j]+1, :]
    end
    #Danielson-Lanczos algorithm
    # Z = complex.(interval(z2))
    Z = z2
    Index = reshape([1:n*col;], n, col)

    theta = sign * (0:n-1) / n # division exact because n is power of 2
    itheta = interval(T, theta)
    Phi = complex.(cospi.(itheta), sinpi.(itheta)) # SLOW?
    # Phi = cospi.(theta) + im*sinpi.(theta)

    v = [1:2:n;]
    w = [2:2:n;]
    t = Z[w, :]
    Z[w, :] = Z[v, :] - t
    Z[v, :] = Z[v, :] + t
    for index in 1:(log2n-1)
        m = 2^index
        m2 = 2 * m
        vw = reshape([1:n;], m2, Int(n / m2))
        v = vw[1:m, :]
        w = vw[m+1:m2, :]
        indexv = reshape(Index[v[:], :], m, Int(col * n / m2))
        indexw = reshape(Index[w[:], :], m, Int(col * n / m2))
        Phi1 = repeat(Phi[1:Int(n / m):end], outer=[1, Int(col * n / m2)])
        t = Phi1 .* Z[indexw]
        Z[indexw] = Z[indexv] - t
        Z[indexv] = Z[indexv] + t
    end
    reverse(Z[2:end, :], dims=2)
    if sign == -1
        Z = Z / interval(n)
    end
    if isrow_
        Z = transpose(Z) #transpose of Z
    end
    if array1
        Z = Z[:, 1]
    end
    return Z
end
# 
verifyfft(z::Vector{Interval{T}}, sign=1) where {T<:Real} = verifyfft(complex.(z), sign)
# 
# Derive Two-sided Chebyshev coefficients
# M must be power of 2
function interval_chebcoeffs(f, M, I=[-1, 1])
    a = I[1]
    b = I[2]
    n = M - 1
    cpts = interval_chebpts(n, a, b)
    fvals = f.(cpts)
    FourierCoeffs = real(verifyfft([reverse(fvals); fvals[2:end-1]])) # the length of this must be power of 2
    ChebCoeffs = FourierCoeffs[1:n+1] / interval(n)
    ChebCoeffs[1] = ChebCoeffs[1] / interval(2)
    ChebCoeffs[end] = ChebCoeffs[end] / interval(2)
    return ChebCoeffs # return Two-sided Chebyshev
end
# 
function interval_chebcoeffs_complex(f, M, I=[-1, 1])
    a = I[1]
    b = I[2]
    n = M - 1
    cpts = interval_chebpts(n, a, b)
    fvals = f.(cpts)
    FourierCoeffs = verifyfft([reverse(fvals); fvals[2:end-1]]) # the length of this must be power of 2
    ChebCoeffs = FourierCoeffs[1:n+1] / interval(n)
    ChebCoeffs[1] = ChebCoeffs[1] / interval(2)
    ChebCoeffs[end] = ChebCoeffs[end] / interval(2)
    return ChebCoeffs # return Two-sided Chebyshev
end
# 
function feval_parallel(f, t)
    # fval = similar(t)
    fval = (zeros(length(t)))
    Threads.@threads for i = eachindex(t)
        tmp = f.(@view t[i])
        # fval[i] = abs(complex(sup(real(tmp)), sup(imag(tmp))))
        fval[i] = sup(abs(tmp))
    end
    return fval
end
# 
function truncCheb(ia, M)
    return [ia[1]+interval(0,norm(ia[M+1:end],1);format=:midpoint);ia[2:M]]
    # return [ia[1] + interval(0, norm(ia[M+1:end], 1)); ia[2:M]]
    # return ia 
end
# 
function fzeval(f, rho, divx, divr=divx)
    x = -1.0:divx:1.0
    ix = interval.(x[1:end-1], x[2:end])
    r = 0:divr:1
    ir = interval.(r[1:end-1], r[2:end])
    iz = complex.(cospi.(ix), sinpi.(ix))
    ie = (interval(rho) * iz + interval(1) ./ (interval(rho) * iz)) ./ interval(2)
    iee = ir * transpose(ie)
    fiee = feval_parallel(f, iee)
    return maximum(fiee)
end
# 
function interval_cheb(f, I=[-1, 1]; ϵ=interval(2^-52), divx=2^-3, divr=divx, tolerance=5e-15) # for general func
    # a = cheb(f, I, tol=5e-12)
    a = cheb(f, I, tol=tolerance)
    # Special case odd/even function
    odd_even = 0
    if all(a[2:2:end] .== 0)
        odd_even = 1 # even function: 1
    elseif all(a[1:2:end] .== 0)
        odd_even = -1 #  odd function: -1
    end
    M = length(a) # Set M
    M̃ = nextpow(2, M) + 1 # Set M̃
    ia = interval_chebcoeffs(f, M̃, I) # Coeffs of p̃(x)
    # Special case odd/even function
    if odd_even == 1 # even function
        ia[2:2:end] .= interval(0)
    elseif odd_even == -1 # odd function
        ia[1:2:end] .= interval(0)
    end
    # Truncation error is in the zero mode
    ia = truncCheb(ia, M) # Coeffs of Πₘp̃(x)
    # Set rho of Bernstein ellipse
    rho = ϵ^(-interval(1) / (interval(M̃) - interval(1)))
    I1 = interval(I[1])
    I2 = interval(I[2])
    i1 = interval(1)
    i2 = interval(2)
    g(ξ) = f((i1 - ξ) / i2 * I1 + (i1 + ξ) / i2 * I2)
    fz = fzeval(g, rho, divx, divr) # Evaluate f(z)
    # Interpolation error via Bernstein ellipse is also in the zero mode
    err = (interval(4) * rho^(-(interval(M̃) - interval(1))) / (rho - interval(1))) * interval(fz)
    # midrad form of interval Cheb interpolation
    ia[1] = ia[1] + interval(0, err; format=:midpoint)
    return ia
end
# 
function biginterval_cheb(f, I=[-1, 1]; ϵ=interval(eps(BigFloat)), div=2^-3, tolerance=5e-23) # for general func
    # a = cheb(f, I, tol=5e-12)
    a = bigcheb(f, I, tol=tolerance)
    # Special case odd/even function
    odd_even = 0
    if all(a[2:2:end] .== 0)
        odd_even = 1 # even function: 1
    elseif all(a[1:2:end] .== 0)
        odd_even = -1 #  odd function: -1
    end
    M = length(a) # Set M
    M̃ = nextpow(2, M) + 1 # Set M̃
    ia = interval_chebcoeffs(f, big(M̃), I) # Coeffs of p̃(x)
    # Special case odd/even function
    if odd_even == 1 # even function
        ia[2:2:end] .= interval(0)
    elseif odd_even == -1 # odd function
        ia[1:2:end] .= interval(0)
    end
    # Truncation error is in the zero mode
    ia = truncCheb(ia, M) # Coeffs of Πₘp̃(x)
    # Set rho of Bernstein ellipse
    rho = ϵ^(-interval(1) / (interval(M̃) - interval(1)))
    I1 = interval(I[1])
    I2 = interval(I[2])
    i1 = interval(1)
    i2 = interval(2)
    g(ξ) = f((i1 - ξ) / i2 * I1 + (i1 + ξ) / i2 * I2)
    fz = fzeval(g, rho, div) # Evaluate f(z)
    # Interpolation error via Bernstein ellipse is also in the zero mode
    err = (interval(4) * rho^(-(interval(M̃) - interval(1))) / (rho - interval(1))) * interval(fz)
    # midrad form of interval Cheb interpolation
    ia[1] = ia[1] + interval(0, err; format=:midpoint)
    return ia
end
# 
function interval_cheb_complex(f, I=[-1, 1]; ϵ=interval(2^-52), div=2^-3) # for general func
    a = cheb_complex(f, I)
    M = length(a) # Set M
    M̃ = nextpow(2, M) + 1 # Set M̃
    ia = interval_chebcoeffs_complex(f, M̃, I) # Coeffs of p̃(x)
    # Truncation error is in the zero mode
    ia = truncCheb(ia, M) # Coeffs of Πₘp̃(x)
    # Set rho of Bernstein ellipse
    rho = ϵ^(-interval(1) / (interval(M̃) - interval(1)))
    I1 = interval(I[1])
    I2 = interval(I[2])
    i1 = interval(1)
    i2 = interval(2)
    g(ξ) = f((i1 - ξ) / i2 * I1 + (i1 + ξ) / i2 * I2)
    fz = fzeval(g, rho, div) # Evaluate f(z)
    # Interpolation error via Bernstein ellipse is also in the zero mode
    err = (interval(4) * rho^(-(interval(M̃) - interval(1))) / (rho - interval(1))) * interval(fz)
    # midrad form of interval Cheb interpolation
    ia[1] = ia[1] + interval(0, err; format=:midpoint)
    return ia
end
# 
function midrad!(ia)
    ia[1] = interval(mid(ia[1]), sum(interval(radius.(ia))); format=:midpoint)
    ia[2:end] = interval(mid.(ia[2:end]))
    # return ia
end
# 
function chebint(ia::Vector{Interval{T}}, I=[-1, 1]) where {T<:Real} # Input is Two-sided
    midrad!(ia) # Transform ia to midrad form
    M = length(ia)
    n = interval(Vector(0:2:M-1))
    # @show sum(2*a[1:2:end]./(1.0 .- n.^2))*((I[2]-I[1])/2)
    i2 = interval(2.0)
    i1 = interval(1.0)
    return sum(i2 * ia[1:2:end] ./ (i1 .- n .^ i2)) * ((interval(I[2]) - interval(I[1])) / i2)
end
# 
function chebint(ia::Vector{Complex{Interval{T}}}, I=[-1, 1]) where {T<:Real} # Input is Two-sided
    midrad!(ia) # Transform ia to midrad form
    M = length(ia)
    n = interval(Vector(0:2:M-1))
    # @show sum(2*a[1:2:end]./(1.0 .- n.^2))*((I[2]-I[1])/2)
    i2 = interval(2.0)
    i1 = interval(1.0)
    return sum(i2 * ia[1:2:end] ./ (i1 .- n .^ i2)) * ((interval(I[2]) - interval(I[1])) / i2)
end
# 
function chebindefint(ia::Vector{Interval{T}}, I=[-1, 1]) where {T<:Real} # Input is Two-sided (inverval)
    M = length(ia)
    ia_ext = zeros(Interval{T}, M + 2)
    ia_ext[1] = interval(2) * ia[1]
    ia_ext[2:M] = ia[2:M]
    iA = zeros(Interval{T}, M + 1)
    for n = 1:M
        iA[n+1] = (ia_ext[n] - ia_ext[n+2]) / interval(2n)
    end
    iA[1] = sum(iA[2:2:end]) - sum(iA[3:2:end]) # takes the value 0 at the left endpoint
    return iA * (interval(I[2]) - interval(I[1])) / interval(2)
end
# 
function eval_interval_cheb(a::Vector{Interval{T}}, x, I=[-1, 1]) where {T<:Real}
    x = interval(T,x)
    M = length(a) # M: size of chebyshev
    I1 = interval(T,I[1])
    I2 = interval(T,I[2])
    k = interval(T,Vector(0:M-1))
    ξ = interval(T,2) * (x .- I1) / (I2 - I1) .- interval(T,1)
    return (cos.(k' .* acos.(ξ))*a)[1]
end
# 
import IntervalArithmetic: exp, sin, cos
function exp(x::Complex{Interval{T}}) where {T<:Real}
    xreal = real(x)
    ximag = imag(x)
    return exp(xreal) * complex(cos(ximag), sin(ximag))
end
# 
function sin(x::Complex{Interval{T}}) where {T<:Real} # z = a + b * im, iz = -b + a * im, -iz = b - a * im
    return (exp(interval(im) * x) - exp(-interval(im) * x)) / interval(2im)
    # tmp = (cexp((-b,a)) .- cexp((b,-a))) # exp(iz) - exp(-iz)
    # return  [tmp[2], -tmp[1]] / interval(2) # z / (2i)
end
# 
function cos(x::Complex{Interval{T}}) where {T<:Real} # z = a + b * im, iz = -b + a * im, -iz = b - a * im
    return (exp(interval(im) * x) + exp(-interval(im) * x)) / interval(2)
    # tmp = (cexp((-b,a)) .- cexp((b,-a))) # exp(iz) - exp(-iz)
    # return  [tmp[2], -tmp[1]] / interval(2) # z / (2i)
end
# 
integrant(x) = exp(-x^2) #erf関数の被積分関数: 2 / sqrt(π)は後でかける
integrant(x::Interval{T}) where {T<:Real} = exp(-x^interval(2)) #*(interval(2) / sqrt(interval(π)))
integrant(x::Complex{Interval{T}}) where {T<:Real} = exp(-x * x)
# f(x) = 2 * exp(-x^2) / sqrt(π) #erf関数の被積分関数
# f(x::Interval{T}) where {T<:Real} = interval(2) * exp(-x^interval(2)) / sqrt(interval(π))
# function f(x::Complex{Interval{T}}) where {T<:Real}
#     x2 = -x * x
#     x2r = real(x2)
#     x2i = imag(x2)
#     return interval(2) * exp(x2r) * complex(cos(x2i), sin(x2i)) / sqrt(interval(π)) # 2*exp(-x^2)/sqrt(π)
# end
dom = [0.0, 5.864]
ia = interval_cheb(integrant, dom, ϵ=interval(2^-67))
iA = chebindefint(ia, dom)
function erf_point(x::Float64, dom, iA)
    if x > 5.864
        return interval(1-2^-53,1)
    elseif x < -5.864
        return interval(-1, -1+2^-53)
    # elseif dom[2] < x
    #     # dom = [0, x]
    #     # iA = chebindefint(interval_cheb(f, dom), I=dom)
    #     # return eval_interval_cheb(iA, x, I=dom) * (interval(2) / sqrt(interval(π)))
    #     dom = [0, x]
    #     ia = interval_cheb(f, dom)
    #     return chebint(ia, dom) * (interval(2) / sqrt(interval(π)))
    # elseif x < -dom[2]
    #     # dom = [0, -x]
    #     # iA = chebindefint(interval_cheb(f, dom), I=dom)
    #     # return -eval_interval_cheb(iA, -x, I=dom) * (interval(2) / sqrt(interval(π)))
    #     dom = [0, -x]
    #     ia = interval_cheb(f, dom)
    #     return chebint(ia, dom) * (interval(2) / sqrt(interval(π)))
    else
        if x == 0
            return 0
        elseif x > 0
            return eval_interval_cheb(iA, x, dom) * (interval(2) / sqrt(interval(π)))
        else
            return -eval_interval_cheb(iA, -x, dom) * (interval(2) / sqrt(interval(π)))
        end
    end
end
# Overload erf function in SpecialFunctions.jl
import SpecialFunctions: erf
erf(x::Interval{T}) where {T<:Real} = interval(erf_point(inf(x), dom, iA), erf_point(sup(x), dom, iA))
ierf(x) = erf_point(convert(Float64,x), dom, iA)
function erf(iz::Complex{Interval{T}}) where {T<:Real}
    z = mid(iz)
    f(x) = exp(-x^2) #erf関数の被積分関数: 2 / sqrt(π)は後でかける
    f(x::Interval{T}) where {T<:Real} = exp(-x^interval(2)) #*(interval(2) / sqrt(interval(π)))
    f(x::Complex{Interval{T}}) where {T<:Real} = exp(-x * x)
    gg(ξ) = f((z / 2) .* (ξ + 1))
    gg(ξ::Interval{T}) where {T<:Real} = f((iz / interval(2)) .* (ξ + interval(1)))
    gg(ξ::Complex{Interval{T}}) where {T<:Real} = f((iz / interval(2)) .* (ξ + interval(1)))
    return (interval(z) / sqrt(interval(π))) * chebint(interval_cheb_complex(gg))
end
ierf(z::Complex{T}) where {T<:Real} = erf(interval(z))
# 
# function chebroots(ia::Vector{Interval{T}}, I=[-1, 1]) where {T<:Real}
#     I_lo = I[1]
#     I_up = I[2]

#     n = length(ia)
#     du = [ones(n - 3) * 0.5; 1] # no error because there elements are power of 2
#     dl = ones(n - 2) * 0.5 # no error because there elements are power of 2
#     d = zeros(n - 1)

#     C_1 = Tridiagonal(dl, d, du)
#     iC_1 = interval(T, C_1)
#     iC_2 = interval(T, zeros(n - 1, n - 1))
#     iC_2[:, 1] = reverse(ia[1:end-1] / ia[n])
#     iC_2 = interval(0.5) * iC_2
#     iC = iC_1 - iC_2
#     C = mid.(iC)

#     lam, x = eigen(convert.(Float64, C))
#     ε = 100 * eps() * (I_up - I_lo) * 0.5
#     ind = findall((-1 - ε .≤ real(lam) .≤ 1 + ε) .& (imag(lam) .≈ 0))
#     lam = real(lam[ind]) # Approximate zeros
#     x = real(x[:, ind]) # Approximate eigenvector
#     ilam = interval(T, zeros(length(lam)))

#     for i = eachindex(lam)
#         ilam[i] = verifyeig(iC, convert(T, lam[i]), convert.(T, x[:, i]))
#     end

#     if I_lo == -1.0 && I_up == 1.0
#         return ilam
#     else
#         return (interval(1.0) .- ilam) .* interval(I_lo) / interval(2) + (interval(1.0) .+ ilam) .* interval(I_up) / interval(2)
#     end
# end
include("IntervalFunctions.jl")
function chebroots(ip::Vector{Interval{T}}, I=[-1, 1]) where {T<:Real}
    I_lo = I[1]; I_up = I[2]
    i1 = interval(1.0); i2 = interval(2.0)

    n = length(ip)
    du = [ones(n - 3) * 0.5; 1] # no error because there elements are power of 2
    dl = ones(n - 2) * 0.5 # no error because there elements are power of 2
    d = zeros(n - 1)

    C_1 = Tridiagonal(dl, d, du)
    iC_1 = interval(T, C_1)
    iC_2 = interval(T, zeros(n - 1, n - 1))
    iC_2[:, 1] = reverse(ip[1:end-1] / ip[n])
    iC_2 = interval(0.5) * iC_2
    iC = iC_1 - iC_2

    X = eigvecs(mid.(iC))
    allroots = verifyalleig(iC, X)
    ε = maximum(radius,real(allroots))
    # ε = 100 * eps() * (I_up - I_lo) * 0.5
    if ε < 1e-3 # maximum tolerance for output
        isinI = -1.0 - ε .<= inf.(real(allroots)) .&& sup.(real(allroots)) .<= 1.0 + ε
        isreR = issubset_interval.(interval(0.0),imag(allroots))
        ind = findall(isinI .&& isreR)
        if I_lo==-1.0 && I_up==1.0
            return real(allroots[ind])
        else
            ix = real(allroots[ind])
            return (i1 .- ix[ind]).* I_lo/i2 + (i1 .+ ix[ind]).*I_up/i2
        end
    else
        return NaN
    end
end
# 
function chebdiff(ia::Vector{Interval{T}}, I=[-1, 1]) where {T} # Input is Two-sided (inverval)
    M = length(ia)
    ib = zeros(Interval{T}, M + 1)
    i2 = interval(T, 2)
    for r = M-1:-1:1
        ib[r] = ib[r+2] + i2 * interval(r) * ia[r+1]
    end
    ib[1] /= interval(2.0)
    return ib[1:end-2] * (interval(2) / (interval(I[2]) - interval(I[1]))) # Output is Two-sided (interval)
end
# 
function endpoints_of_cheb(ia::Vector{Interval{T}}) where {T} # Input is two-sided Chebyshev
    n = length(ia)
    atm1 = dot(interval((-1) .^ (0:n-1)), ia) # endpoint at -1
    at1 = sum(ia) # endpoint at 1
    return [atm1, at1]
end
# 
function chebmax(ia, I=[-1,1]) # Input is two-sided Chebyshev
    I_lo = interval(I[1]); I_up = interval(I[2]); M = length(ia)
    i1 = interval(1.0); i2 = interval(2.0)
    ep = endpoints_of_cheb(ia)
    midrad!(ia)
    ix = chebroots(chebdiff(ia, I))
    k = 0:M-1
    fxc = cos.(interval(Vector(k))' .* acos.(ix)) * ia
    fvals = [ep[1];fxc[1:end];ep[2]]
    ix = [-i1; ix; i1]
    ind = findall(isequal_interval.(fvals,maximum(fvals)))
    if isempty(ind)
        return maximum(fvals)
    else
        return fvals[ind], (i1 .- ix[ind]).* I_lo/i2 + (i1 .+ ix[ind]).*I_up/i2
    end
end
# 
function chebmin(ia, I=[-1, 1]) # Input is two-sided Chebyshev
    I_lo = interval(I[1]); I_up = interval(I[2]); M = length(ia)
    i1 = interval(1.0); i2 = interval(2.0)
    ep = endpoints_of_cheb(ia)
    midrad!(ia)
    ix = chebroots(chebdiff(ia, I))
    k = 0:M-1
    fxc = cos.(interval(Vector(k))' .* acos.(ix)) * ia
    fvals = [ep[1]; fxc[1:end]; ep[2]]
    ix = [-i1; ix; i1]
    ind = findall(isequal_interval.(fvals, minimum(fvals)))
    if isempty(ind)
        return minimum(fvals)
    else
        return fvals[ind], (i1 .- ix[ind]).* I_lo/i2 + (i1 .+ ix[ind]).*I_up/i2
    end
end