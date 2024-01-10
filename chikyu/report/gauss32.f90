program gauss
  implicit none
  REAL, dimension(4,4) :: a, b
  REAL, dimension(1,4) :: c, x
  REAL, dimension(4) :: tmp
  REAL :: ans, maxnum, maxv, temp
  INTEGER :: i, j, k, n = 4
  a = reshape( (/10, 1, 4, 0, 1, 10, 5, -1, 4, 5, 10, 7, 0, -1, 7, 9/), (/4, 4/) )
  c = reshape( (/15, 15, 26, 15/), (/1,4/) )  


  do i = 1, n, 1
  ! a is matrix, 
  ! scaling
    maxnum = MAXVAL(abs(a(:, i)))
    a(:, i) = a(:, i) / maxnum
    c(:, i) = c(:, i) / maxnum
  end do


  do i = 1, n-1, 1

    ! pivote
    ! find max value in row
    maxv = a(i, i)
    k = i
    do j = i, n, 1
      if (maxv >= a(i, j)) then
        maxv = maxv
      else
        maxv = a(i, j)
        k = j
      end if
    end do
    ! swap row
    tmp = a(:, i)
    a(:, i) = a(:, k)
    a(:, k) = tmp
    temp = c(1, i)
    c(1, i) = c(1, k)
    c(1, k) = temp

    ! delete
    if (a(i,i) /= 0.0 .and. i /= 4) then
      do j = i + 1, n
        if (a(i,i) /= 0.0 .and. a(i,j) /= 0.0) then
          b(i, j) = -a(i, j) / a(i, i)
          a(:,j) = a(:, j) - (a(i, j)/a(i, i))*a(:,i)
          c(:,j) = c(:, j) + b(i, j)*c(:,i)
        end if
      end do
    end if
  end do
  do i = 1, n, 1
    print *, a(:,i)
  end do
  
  do i = n, 1, -1
    x(1,i) = (c(1,i) - SUM((/ (a(j, i) * x(1,j) , j = i, n) /))) / a(i,i)
  end do

  do i = 1, n, 1
    print *, x(:,i)
  end do
end program gauss
