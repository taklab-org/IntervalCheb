### Interval Matrix Multiplication
function ufp(P)
    u = 2.0^(-53);
    ϕ = 2.0^52 + 1;
    q = ϕ * P;
    T = (1 - u)*q;
    return abs(q - T);
end

function succ(c)
    s_min = 2.0^-1074;
    u = 2.0^-53;
    ϕ = u * (1.0 + 2.0 * u);
    if abs(c) >= (1. / 2.) * u^(-2) * s_min # 2^(-969)(Float64)
        e = ϕ * abs(c);
        succ = c + e;
    elseif abs(c) < (1. / 2.) * u^(-1) * s_min # 2^(-1022)(Float64)
        succ = c + s_min;
    else
        C = u^(-1) * c;
        e = ϕ * abs(C);
        succ = (C + e) * u;
    end
    return succ
end

function pred(c)
    s_min = 2.0^-1074;
    u = 2.0^-53;
    ϕ = u * (1.0 + 2.0 * u);
    if abs(c) >= (1. / 2.) * u^(-2) * s_min # 2^(-969)(Float64)
        e = ϕ * abs(c);
        pred = c - e;
    elseif abs(c) < (1. / 2.) * u^(-1) * s_min # 2^(-1022)(Float64)
        pred = c - s_min;
    else
        C = u^(-1) * c;
        e = ϕ * abs(C);
        pred = (C - e) * u;
    end
    return pred
end

function mm_ufp(A_mid, B_mid) # A_mid, B_mid: Point matrix
    u = 2.0^(-53);
    realmin = 2.0^(-1022);
    n = size(A_mid,2);

    if(2*(n+2)*u>=1)
        error("mm_ufp is failed!(2(n+2)u>=1)")
    end
    # C_mid = A_mid * B_mid;
    # C_rad = (n+2) * u * ufp.(abs.(A_mid)*abs.(B_mid)) .+ realmin;
    # return C_mid, C_rad;
    return A_mid * B_mid, (n+2) * u * ufp.(abs.(A_mid)*abs.(B_mid)) .+ realmin
end

function imm_ufp(A_mid, A_rad, B_mid, B_rad) # A = <A_mid, A_rad>, B = <B_mid, B_rad>: Interval matrix
    u = 2.0^(-53);
    realmin = 2.0^(-1022);
    n = size(A_mid,2);

    if(2*(n+2)*u>=1)
        error("mm_ufp is failed!(2(n+2)u>=1)")
    end
#     C, R = mm_ufp(A_mid,B_mid);
    # C_mid = A_mid * B_mid;
    R = (n+2) * u * ufp.(abs.(A_mid)*abs.(B_mid)) .+ realmin;

#     T_1, T_2 = mm_ufp(abs.(A_mid), B_rad);
    T1 = abs.(A_mid) * B_rad;
    T2 = (n+2)*u*ufp.(T1) .+ realmin;

#     T_3 = succ.(abs.(B_mid)+B_rad);
    T3 = succ.(abs.(B_mid)+B_rad)

#     T_4, T_5 = mm_ufp(A_r, T_3);
    T4 = A_rad * T3;
    T5 = (n+2)*u*ufp.(T4) .+ realmin;

    rad_sum = R + T1 + T2 + T4 + T5;

    # C_rad = succ.(rad_sum + 4*u*ufp.(rad_sum));

    # return C_mid, C_rad;
    return A_mid * B_mid, succ.(rad_sum + 4*u*ufp.(rad_sum))
end

# USE IntervalArithmetic.jl
using IntervalArithmetic
function int_mul(A::Matrix{T}, B::Matrix{T}) where T
    Cmid, Crad = mm_ufp(A, B);
    return interval(Cmid, Crad; format=:midpoint)
    # return Cmid .± Crad
end

function int_mul(A::Matrix{Interval{T}}, B::Matrix{T}) where T
    Cmid, Crad = imm_ufp(mid.(A), radius.(A), B, zeros(size(B)));
    return interval(Cmid, Crad; format=:midpoint)
    # return Cmid .± Crad
end

function int_mul(A::Matrix{T}, B::Matrix{Interval{T}}) where T
    Cmid, Crad = imm_ufp(A, zeros(size(A)), mid.(B), radius.(B));
    return interval(Cmid, Crad; format=:midpoint)
    # return Cmid .± Crad
end

function int_mul(A::Matrix{Interval{T}}, B::Matrix{Interval{T}}) where T
    Cmid, Crad = imm_ufp(mid.(A), radius.(A), mid.(B), radius.(B));
    return interval(Cmid, Crad; format=:midpoint)
    # return Cmid .± Crad
end

function int_mul(A::Matrix{Complex{T}}, B::Matrix{T}) where T
    Ar = real.(A); Ai = imag.(A); # (Ar + im*Ai)*B = Ar*B + im*(Ai*B)
    return complex.(int_mul(Ar, B), int_mul(Ar, B))
end

function int_mul(A::Matrix{T}, B::Matrix{Complex{T}}) where T
    Br = real.(B); Bi = imag.(B); # A*(Br + im*Bi) = A*Br + im*(A*Bi)
    return complex.(int_mul(A, Br), int_mul(A, Bi))
end

function int_mul(A::Matrix{Complex{T}}, B::Matrix{Complex{T}}) where T
    Ar = real.(A); Ai = imag.(A); Br = real.(B); Bi = imag.(B);
    # (Ar + im*Ai)*(Br + im*Bi) = (Ar*Br - Ai*Bi) + im*(Ar*Bi + Ai*Br)
    return complex.((int_mul(Ar,Br) - int_mul(Ai, Bi)), (int_mul(Ar, Bi) + int_mul(Ai, Br)))
end


### Interval Linear system solver
import LinearAlgebra: opnorm
function LinearAlgebra.opnorm(a::Matrix{Complex{Interval{Float64}}},p::String)
    if p == "1"
        suma = sum(abs.(a), dims = 1)
        return interval(maximum(inf,suma), maximum(sup,suma))
    elseif p == "Inf"
        suma = sum(abs.(a), dims = 2)
        return interval(maximum(inf,suma),maximum(sup,suma))
    end
    return NaN
end

function LinearAlgebra.opnorm(a::Matrix{Interval{Float64}},p::String)
    if p == "1"
        suma = sum(abs.(a), dims = 1)
        return interval(maximum(inf,suma), maximum(sup,suma))
    elseif p == "Inf"
        suma = sum(abs.(a), dims = 2)
        return interval(maximum(inf,suma),maximum(sup,suma))
    end
    return NaN
end

function verifylss_iB(iA,iB) # verify the solution element-wisely
    A = mid.(iA); B = mid.(iB)
    X̄ = A\B
    n = size(X̄,2)
    R = inv(A)
    #########
    iG = interval(Matrix{Float64}(I, n, n)) - interval(R)*iA
    α = opnorm(iG,"Inf")# Interval arithmetic
    #########
    if sup(α) < 1
        η = (abs.(iG)*interval(ones(n)))/(interval(1)-α)
        Err = interval(zeros(n,n))
        X̄ = interval(X̄)
        ir = iA*X̄ - iB # Interval arithmetic
        iRr = interval(R)*ir
        for i = 1:n
            Err[:,i] = abs.(iRr[:,i]) + interval(sup(norm(iRr[:,i],Inf)))*η # Interval arithmetic
        end
        return interval(X̄, Err, format=:midpoint)
    else
        println("Oh my way, verification is failed...")
        return nan
    end
end


### Interval all eigenvalues solver

function verifyalleig(iA, X)
    n = size(iA, 2)
    # iD = diagm(interval(λ))
    iX = interval(X)
    iB = int_mul(iA, iX)
    iG = verifylss_iB(iX, iB)
    ir = interval(zeros(n))
    ic = diag(iG)
    for i = 1:n
        for j = 1:n
            if i != j
                ir[i] += interval(mag(iG[i, j]))
            end
        end
        ir[i] += interval(radius(ic[i]))
    end
    return interval(ic, ir, format=:midpoint)
end

### Interval eigen solver (eigpair)

function verifyeig(iA::Matrix{Interval{T}}, lam, x, B=Matrix{T}(I, size(iA))) where {T<:Real}
    if typeof(lam) <: Complex || eltype(x) <: Complex
        lam = convert(Complex{T}, lam)
        x = convert.(Complex{T}, Vector(x))
    else
        lam = convert(T, lam)
        x = convert.(T, Vector(x))
    end
    x = x ./ norm(x)
    ysize = length(x)
    ilam = interval(lam)
    ix = interval(x)
    iB = interval(B)

    function iDF(w)
        mat = (zeros(typeof(ilam), length(w), length(w)))
        mat[1, 2:end] = transpose(interval(2) * (ix + w[2:end]))
        mat[2:end, 1] = -iB * (ix + w[2:end])
        mat[2:end, 2:end] = iA - (ilam + w[1]) * iB
        return mat
    end
    zero = zeros(T, ysize + 1)
    R = inv(mid.(iDF(zero)))
    iR = interval(R)
    iz = -iR * [dot(ix, ix) - interval(1.0); iA * ix - ilam * iB * ix]
    ϵ = 2 * sup(norm(iz, 1))
    if isreal(lam) && isreal(x)
        lam = real(lam)
        x = real(x)
        id = interval(0, ϵ; format=:midpoint)
        iy = interval.(zeros(ysize), ϵ; format=:midpoint)
        iI = interval(Matrix{T}(I, ysize + 1, ysize + 1))
    else
        id = Complex(interval(0, ϵ; format=:midpoint), interval(0, ϵ; format=:midpoint))
        iy = Complex.(interval.(zeros(ysize), ϵ; format=:midpoint), interval.(zeros(ysize), ϵ; format=:midpoint))
        iI = interval(Matrix{Complex{T}}(I, ysize + 1, ysize + 1))
    end
    iw = [id; iy]
    g(w) = iz + (iI - iR * iDF(w)) * w
    gw = g(iw)
    if all(issubset_interval.(gw, iw)) #gw .⊂ iw
        # while maximum(radius, gw) / norm([lam; x], 1) >= 1e3 * eps(T)
        #     gw = g(gw)
        # end
        return ilam + gw[1] #, ix .+ gw[2:end] #固有ベクトルも返すようにする
    else
        return NaN
    end
end